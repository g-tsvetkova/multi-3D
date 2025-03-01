import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from transformers import OPTConfig, OPTModel, AutoConfig, AutoModelForCausalLM
from models.mesh_xl.tokenizer import MeshTokenizer
from typing import Dict


class MTPMeshXL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = MeshTokenizer(args)
        self.n_future_tokens = 2  # Predict next 2 tokens

        # Token config
        self.vocab_size = 131 #50272
        self.bos_token_id = 2
        self.eos_token_id = 2
        self.pad_token_id = 1

        if args.pretrained_weights:
            config = AutoConfig.from_pretrained(
            args.llm, 
            n_positions=8192,
            max_position_embeddings=8192,
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id
            )
            self.transformer = AutoModelForCausalLM.from_pretrained(
            args.llm, 
            config=config,
            ignore_mismatched_sizes=True
        )
            
        else:
            # Create model config
            self.config = OPTConfig(
                vocab_size=self.vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                max_position_embeddings=7300,
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
        
        # # Initialize weights
        # self._init_weights()
        
        # Set to training mode
        self.train()

    # def _init_weights(self):
    #     """Initialize weights with small values to start training from scratch"""
    #     def _init_layer(module):
    #         if isinstance(module, (nn.Linear, nn.Embedding)):
    #             module.weight.data.normal_(mean=0.0, std=0.02)
    #             if isinstance(module, nn.Linear) and module.bias is not None:
    #                 module.bias.data.zero_()
    #         elif isinstance(module, nn.LayerNorm):
    #             module.bias.data.zero_()
    #             module.weight.data.fill_(1.0)

    #     self.apply(_init_layer)

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

    def freeze_layers(self):
        # Freeze entire trunk initially
        for param in self.trunk.parameters():
            param.requires_grad = False

        # Unfreeze last N layers of trunk
        num_unfrozen_layers = 1  # Customize this
        for layer in self.trunk.decoder.layers[-num_unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Ensure offset embeddings are trainable
        for param in self.offset_embeddings.parameters():
            param.requires_grad = True
        
        # Heads and offset embeddings remain trainable (their requires_grad is True by default)
        
        # Count and print trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Percentage Trainable: {trainable_params/total_params:.2%}")
        
        # Print trainable component names
        print("\nTrainable components:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  {name}")
        
        return self
    
    def count_parameters(self):
        """Print and return the number of parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total Parameters: {total_params:,}')
        print(f'Trainable Parameters: {trainable_params:,}')
        return total_params, trainable_params

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
        offset_emb = self.offset_embeddings(
            torch.tensor(0, device=trunk_out.device, dtype=torch.long)
        )
        features = trunk_out + offset_emb[None, None, :]
        logits = head(features)

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
            offset_emb = self.offset_embeddings(
                torch.tensor(0, device=trunk_out.device, dtype=torch.long)
            )
            features = trunk_out + offset_emb[None, None, :]
            logits = self.heads[0](features[:, -1:])

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

def test_mtp_model(model):
    # Set model to eval mode
    model.eval()
    
    # Create a small test batch
    test_dict = {
        "input_ids": torch.randint(0, model.tokenizer.codebook_size, (2, 10)),  # batch_size=2, seq_len=10
        "attention_mask": torch.ones(2, 10)  # All tokens are valid
    }
    
    print("Testing perplexity calculation...")
    try:
        with torch.no_grad():
            perp_output = model(test_dict, is_eval=True, is_generate=False)
        print("✓ Perplexity calculation successful")
        print(f"Perplexity value: {perp_output['perplexity']:.2f}")
    except Exception as e:
        print("✗ Perplexity calculation failed")
        print(f"Error: {str(e)}")
    
    print("\nTesting generation...")
    try:
        with torch.no_grad():
            gen_output = model(
                test_dict,
                is_eval=True,
                is_generate=True,
                max_length=15,
                num_return_sequences=2
            )
        print("✓ Generation successful")
        print("Generated sequences shape:", gen_output["input_ids"].shape)
    except Exception as e:
        print("✗ Generation failed")
        print(f"Error: {str(e)}")

# Usage:
# model = MTPMeshXL(args)  # Your model instance
# test_mtp_model(model)
