import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from transformers import OPTConfig, OPTForCausalLM
from models.mesh_xl.tokenizer import MeshTokenizer
from typing import Dict, Optional

class MeshXL(nn.Module):
   
    def train(self, mode: bool = True):
        super().train(mode)
        return self
   
    def __init__(self, args):
        super().__init__()
       
        # Initialize tokenizer
        self.tokenizer = MeshTokenizer(args)
       
        # Speculative decoding parameters
        self.speculative_tokens = 3  # number of tokens predicted at once
        self.speculative_temperature = 1.0
       
        # Causal LM model initialization
        self.vocab_size = self.tokenizer.codebook_size + 3
        self.bos_token_id = self.tokenizer.codebook_size
        self.eos_token_id = self.tokenizer.codebook_size + 1
        self.pad_token_id = self.tokenizer.codebook_size + 2
       
        # Create an OPT model configuration
        config = OPTConfig(
            vocab_size=self.vocab_size,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=2,
            max_position_embeddings=8192,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            activation_function="relu",
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            layerdrop=0.0,
            use_cache=True,
            torch_dtype=torch.float16,
            ffn_dim=1024
        )
       
        config.word_embed_proj_dim = config.hidden_size
        self.transformer = OPTForCausalLM(config)
       
        # A head to predict multiple tokens at once (for speculation)
        self.speculative_head = nn.Linear(
            config.hidden_size,
            self.vocab_size * self.speculative_tokens
        )
       
        self.train()

    def loss_wrapper(self, loss: Tensor) -> Tensor:
        # A placeholder wrapper; can add additional regularization if needed
        for param in self.parameters():
            loss += 0 * torch.sum(param ** 2)
        return loss

    def forward(
        self,
        data_dict: dict = None,
        is_eval: bool = False,
        is_generate: bool = False,
        num_return_sequences: int = 8,
        generation_config: Dict = None
    ) -> dict:
       
        # For evaluation generation calls, go to speculative generation
        if is_eval and is_generate:
            return self.speculative_generate(
                data_dict=data_dict,
                num_return_sequences=num_return_sequences,
                generation_config=generation_config or {}
            )
       
        raise NotImplementedError('Only speculative generation is implemented here.')

    @torch.no_grad()
    def speculative_generate(
        self,
        data_dict: dict = None,
        num_return_sequences: int = 8,
        generation_config: dict = None
    ) -> dict:
        """
        Custom speculative decoding:
        - Start from BOS if no input provided
        - Generate 'speculative_tokens' at a time
        - Apply top-k & top-p filtering
        - Continue until EOS or max_length
        - Detokenize to produce 'recon_faces'
        """
        generation_config = generation_config or {}
        device = next(self.parameters()).device
        max_length = generation_config.get('max_length', 8192)

        if data_dict is None or 'input_ids' not in data_dict:
            input_ids = torch.full(
                (num_return_sequences, 1),
                self.bos_token_id,
                device=device,
                dtype=torch.long
            )
        else:
            input_ids = data_dict['input_ids'].to(device)

        # Generate until max_length or EOS is encountered
        while input_ids.size(1) < max_length:
            # Predict a block of speculative tokens
            speculative_tokens = self._compute_speculative_tokens(input_ids, generation_config)
            input_ids = torch.cat([input_ids, speculative_tokens], dim=1)
           
            if self._should_stop_generation(input_ids):
                break

        # Postprocess and detokenize
        output_ids = self._postprocess_generated_ids(input_ids)
        decoder_output = self.tokenizer.detokenize(input_ids=output_ids)

        # Make sure decoder_output has 'recon_faces' key as expected by the evaluator
        return decoder_output

    def _compute_speculative_tokens(
        self,
        input_ids: Tensor,
        generation_config: Dict
    ) -> Tensor:
        """
        Compute a batch of speculative tokens from the model.
        Request hidden_states and take the last one as last_hidden_state.
        """
        transformer_outputs = self.transformer(
            input_ids,
            output_hidden_states=True
        )

        # The last element of hidden_states is the last hidden state
        last_hidden_state = transformer_outputs.hidden_states[-1]
        last_hidden = last_hidden_state[:, -1:]  # shape: (batch, 1, hidden_size)

        speculative_logits = self.speculative_head(last_hidden)  # (batch, 1, vocab_size*speculative_tokens)
        speculative_logits = speculative_logits.view(
            input_ids.size(0),
            self.speculative_tokens,
            self.vocab_size
        )
       
        # Apply temperature (if needed)
        temperature = generation_config.get('temperature', 1.0)
        if temperature != 1.0:
            speculative_logits = speculative_logits / temperature

        # Filter and sample tokens
        top_k = generation_config.get('top_k', 50)
        top_p = generation_config.get('top_p', 0.95)
        filtered_logits = self._filter_logits(speculative_logits, top_k, top_p)
       
        token_probs = torch.softmax(filtered_logits, dim=-1)
        sampled_tokens = torch.multinomial(token_probs.view(-1, token_probs.size(-1)), 1)
        sampled_tokens = sampled_tokens.view(input_ids.size(0), -1)

        return sampled_tokens

    def _filter_logits(
        self,
        logits: Tensor,
        top_k: int,
        top_p: float
    ) -> Tensor:
        """
        Apply top-k and top-p filtering on the logits for speculative tokens.
        logits shape: (batch, speculative_tokens, vocab_size)
        """
        filtered_logits = logits.clone()

        # Top-K Filtering
        if top_k > 0:
            for i in range(filtered_logits.size(0)):
                for j in range(filtered_logits.size(1)):
                    row = filtered_logits[i, j]
                    if row.size(0) > top_k:
                        top_k_values, top_k_indices = torch.topk(row, top_k)
                        mask = torch.full_like(row, float('-inf'))
                        mask.scatter_(0, top_k_indices, top_k_values)
                        filtered_logits[i, j] = mask
       
        # Top-P (Nucleus) Filtering
        if top_p < 1.0:
            for i in range(filtered_logits.size(0)):
                for j in range(filtered_logits.size(1)):
                    row = filtered_logits[i, j]
                    sorted_logits, sorted_indices = torch.sort(row, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
                    nucleus_mask = cumulative_probs > top_p
                    if nucleus_mask.all():
                        nucleus_mask[0] = False
                    # Replace beyond threshold with -inf
                    sorted_logits[nucleus_mask] = float('-inf')
                    filtered_logits[i, j].scatter_(0, sorted_indices, sorted_logits)

        return filtered_logits

    def _should_stop_generation(self, input_ids: Tensor) -> bool:
        """
        Determine if generation should stop (if EOS token found or max length reached).
        """
        if input_ids.size(1) >= 8192:
            return True
        # If any sequence ended with EOS
        if torch.any(input_ids[:, -1] == self.eos_token_id):
            return True
        return False

    # def _postprocess_generated_ids(self, input_ids: Tensor) -> Tensor:
    #     """
    #     Post-process generated ids by removing BOS token and replacing EOS with PAD.
    #     """
    #     output_ids = input_ids[:, 1:]  # remove BOS
    #     output_ids[output_ids == self.eos_token_id] = self.tokenizer.pad_id
    #     return output_ids
    
    def _postprocess_generated_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        output_ids = input_ids[:, 1:]  # Remove BOS
        output_ids[output_ids == self.eos_token_id] = self.tokenizer.pad_id
    
        # Ensure divisibility by c=9 (assuming c=9 is a known constant from tokenizer)
        c = 9
        length = output_ids.size(1)
        remainder = length % c
        if remainder != 0:
            # Truncate the extra tokens
            output_ids = output_ids[:, :length - remainder]

        return output_ids

def get_model(args):
    model = MeshXL(args)
    return model