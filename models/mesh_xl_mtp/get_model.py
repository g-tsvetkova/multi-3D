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

        # Shared trunk
        config = OPTConfig(
            vocab_size=self.vocab_size,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            max_position_embeddings=8192,
            ffn_dim=1024,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        self.trunk = OPTModel(config)

        # Prediction heads with transformer layers
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.TransformerDecoderLayer(
                        d_model=config.hidden_size,
                        nhead=config.num_attention_heads,
                        dim_feedforward=config.ffn_dim,
                        batch_first=True,
                    ),
                    nn.Linear(config.hidden_size, self.vocab_size),
                )
                for _ in range(self.n_future_tokens)
            ]
        )

        # Offset embeddings
        self.offset_embeddings = nn.Embedding(self.n_future_tokens, config.hidden_size)

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
        input_ids[:, 0] = self.bos_token_id
        eos_pos = attention_mask.sum(1, keepdim=True) - 1
        input_ids.scatter_(1, eos_pos.long(), self.eos_token_id)

        # Shared trunk processing
        trunk_out = self.trunk(
            input_ids=input_ids.long(), attention_mask=attention_mask
        ).last_hidden_state

        # Multi-token prediction
        total_loss = 0
        for head_idx, head in enumerate(self.heads):
            offset = head_idx + 1

            # Offset-aware features
            offset_emb = self.offset_embeddings(
                torch.tensor(head_idx, device=trunk_out.device)
            )
            features = trunk_out + offset_emb[None, None, :]

            # Generate predictions
            causal_mask = torch.triu(
                torch.ones(
                    trunk_out.size(1), trunk_out.size(1), device=trunk_out.device
                )
                * float("-inf"),
                diagonal=1,
            )
            head_out = head[0](features, features, tgt_mask=causal_mask)
            logits = head[1](head_out)

            # Calculate loss for this head
            targets = input_ids[:, offset : offset + trunk_out.size(1)]
            valid_positions = attention_mask[:, offset : offset + trunk_out.size(1)]

            loss = nnf.cross_entropy(
                logits.permute(0, 2, 1),
                targets,
                ignore_index=self.pad_token_id,
                reduction="none",
            )
            loss = (loss * valid_positions).sum() / valid_positions.sum()
            total_loss += loss

        data_dict["loss"] = total_loss / self.n_future_tokens
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
