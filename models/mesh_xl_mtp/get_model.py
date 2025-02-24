import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from transformers import OPTConfig, OPTModel
from models.mesh_xl.tokenizer import MeshTokenizer
from typing import Dict


class MTPMeshXL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = MeshTokenizer(args)
        self.n_future_tokens = 2  # Predict next 2 tokens

        # Token config
        self.vocab_size = self.tokenizer.codebook_size + 3
        self.bos_token_id = self.tokenizer.codebook_size
        self.eos_token_id = self.tokenizer.codebook_size + 1
        self.pad_token_id = self.tokenizer.codebook_size + 2

        # Create model config
        self.config = OPTConfig(
            vocab_size=self.vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=8000,
            ffn_dim=1024,
            activation_function="relu",
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            layerdrop=0.0,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        
        # Initialize trunk
        self.trunk = OPTModel(self.config)
        
        # Initialize prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.TransformerEncoderLayer(
                    d_model=self.config.hidden_size,
                    nhead=self.config.num_attention_heads,
                    dim_feedforward=self.config.ffn_dim,
                    dropout=self.config.dropout,
                    activation="relu",
                    batch_first=True
                ),
                nn.LayerNorm(self.config.hidden_size),
                nn.Linear(self.config.hidden_size, self.config.vocab_size)
            )
            for _ in range(self.n_future_tokens)
        ])

        # Initialize offset embeddings
        self.offset_embeddings = nn.Embedding(self.n_future_tokens, self.config.hidden_size)
        
        # Initialize weights
        self._init_weights()
        
        # Set to training mode
        self.train()

    def _init_weights(self):
        """Initialize weights with small values to start training from scratch"""
        def _init_layer(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(_init_layer)

    def forward(
        self,
        data_dict: dict = None,
        is_eval: bool = False,
        is_generate: bool = False,
        num_return_sequences: int = 8,
        generation_config: Dict = dict(
            do_sample=True,
            top_k=50,
            top_p=0.95,
        ),
    ) -> dict:

        if not is_eval:
            return self.train_one_step(data_dict)

        if is_eval and not is_generate:
            return self.perplexity(data_dict)

        if is_eval and is_generate:
            return self.generate(
                data_dict=data_dict,
                num_return_sequences=num_return_sequences,
                generation_config=generation_config,
            )

        raise NotImplementedError("Invalid mode combination")

    def train_one_step(self, data_dict: dict) -> dict:
        data_dict = self.tokenizer.tokenize(data_dict)
        input_ids = data_dict["input_ids"]
        attention_mask = data_dict["attention_mask"]

        # Prepare inputs with bos/eos
        input_ids[input_ids == self.tokenizer.pad_id] = self.pad_token_id
        input_ids[:, 0] = self.bos_token_id
        eos_pos = attention_mask.sum(1, keepdim=True) - 1
        input_ids.scatter_(1, eos_pos.long(), self.eos_token_id)

        # Get trunk features
        trunk_outputs = self.trunk(
            input_ids=input_ids.long(),
            attention_mask=attention_mask,
            return_dict=True
        )
        trunk_features = trunk_outputs.last_hidden_state

        # Multi-token prediction
        total_loss = 0
        seq_length = trunk_features.size(1)
        
        for head_idx, head in enumerate(self.heads):
            offset = head_idx + 1
            
            # Add guard condition for sequence length
            if offset >= seq_length:
                continue  # Skip heads that would cause empty tensors
            
            # Add offset embedding
            offset_emb = self.offset_embeddings(
                torch.tensor(head_idx, device=trunk_features.device, dtype=torch.long)  # Explicit dtype
            )
            features = trunk_features + offset_emb[None, None, :]
            
            # Get predictions for valid sequence positions
            logits = head(features[:, :seq_length-offset])  # Only process valid positions
            
            # Get targets with proper bounds checking
            targets = input_ids[:, offset:offset+seq_length-offset]
            target_mask = attention_mask[:, offset:offset+seq_length-offset]
            
            # Calculate loss only if there are valid targets
            if targets.numel() == 0:
                continue

            loss = nnf.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=self.pad_token_id,
                reduction='none'
            )
            
            # Apply mask and average only if there are valid elements
            if target_mask.sum().item() > 0:
                loss = (loss.view_as(targets) * target_mask).sum() / target_mask.sum()
                total_loss += loss

        data_dict["loss"] = total_loss / max(1, self.n_future_tokens)  # Prevent division by zero
        return data_dict


    @torch.no_grad()
    def perplexity(self, data_dict: dict) -> dict:
        # Use first head for perplexity calculation
        data_dict = self.tokenizer.tokenize(data_dict)
        input_ids = data_dict["input_ids"]
        attention_mask = data_dict["attention_mask"]

        # Prepare inputs
        input_ids[:, 0] = self.bos_token_id
        eos_pos = attention_mask.sum(1, keepdim=True) - 1
        input_ids.scatter_(1, eos_pos.long(), self.eos_token_id)

        # Trunk processing
        trunk_out = self.trunk(
            input_ids=input_ids.long(), attention_mask=attention_mask
        ).last_hidden_state

        # First head processing
        head = self.heads[0]
        causal_mask = torch.triu(
            torch.ones(trunk_out.size(1), trunk_out.size(1), device=trunk_out.device)
            * float("-inf"),
            diagonal=1,
        )
        head_out = head[0](trunk_out, trunk_out, tgt_mask=causal_mask)
        logits = head[1](head_out)

        # Calculate perplexity
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = nnf.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_token_id,
        )

        data_dict["perplexity"] = torch.exp(loss)
        return data_dict

    @torch.no_grad()
    def generate(self, data_dict: dict, **kwargs) -> dict:
        # Autoregressive generation with first head
        net_device = next(self.parameters()).device
        input_ids = data_dict["input_ids"].to(net_device)
        max_length = kwargs.get("max_length", 512)

        for _ in range(max_length):
            # Trunk processing
            trunk_out = self.trunk(input_ids).last_hidden_state

            # First head prediction
            causal_mask = torch.triu(
                torch.ones(input_ids.size(1), input_ids.size(1), device=net_device)
                * float("-inf"),
                diagonal=1,
            )
            head_out = self.heads[0][0](trunk_out, trunk_out, tgt_mask=causal_mask)
            logits = self.heads[0][1](head_out[:, -1:])

            # Sampling
            probs = nnf.softmax(logits / kwargs.get("temperature", 1.0), dim=-1)
            next_tokens = torch.multinomial(probs.squeeze(1), num_samples=1)
            input_ids = torch.cat([input_ids, next_tokens], dim=1)

        # Post-processing
        input_ids[input_ids == self.eos_token_id] = self.tokenizer.pad_id
        return self.tokenizer.detokenize(input_ids)

    def loss_wrapper(self, loss: Tensor) -> Tensor:
        # Original regularization (keep unchanged)
        for param in self.parameters():
            loss += 0 * torch.sum(param**2)
        return loss


def get_model(args):
    return MTPMeshXL(args)
