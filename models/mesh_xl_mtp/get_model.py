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
        
        # Tokenizer initialization
        self.tokenizer = MeshTokenizer(args)
        
        # Speculative decoding configuration
        self.speculative_tokens = 3
        self.speculative_temperature = 1
        
        # Causal LM model initialization
        self.vocab_size = self.tokenizer.codebook_size + 3
        self.bos_token_id = self.tokenizer.codebook_size
        self.eos_token_id = self.tokenizer.codebook_size + 1
        self.pad_token_id = self.tokenizer.codebook_size + 2
        
        # OPT Configuration (mostly kept the same as your original)
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
        
        # Add speculative head for multi-token prediction
        self.speculative_head = nn.Linear(
            config.hidden_size, 
            self.vocab_size * self.speculative_tokens
        )
        
        self.train() 

    def forward(
        self, 
        data_dict: dict = None, 
        is_eval: bool = False, 
        is_generate: bool = False,
        num_return_sequences: int = 8,
        generation_config: Dict = None
    ) -> dict:
        # Expanded forward method to handle speculative decoding
        if is_generate:
            # Use an enhanced generation method
            return self.speculative_generate(
                data_dict, 
                num_return_sequences, 
                generation_config or {}
            )
        
        # Existing forward logic for training and evaluation
        return super().forward(data_dict, is_eval, is_generate)

    def train_one_step(self, data_dict: dict) -> dict:
        # Modified training step to support multi-token prediction
        data_dict = self.tokenizer.tokenize(data_dict)
        
        input_ids = data_dict['input_ids']
        attention_mask = data_dict['attention_mask']
        
        # Prepare input with special tokens
        input_ids[input_ids == self.tokenizer.pad_id] = self.pad_token_id
        input_ids[:, 0] = self.bos_token_id
        
        # Generate targets for multiple tokens
        target = self._prepare_multi_token_targets(input_ids, attention_mask)
        
        # Forward pass with potential speculative prediction
        outputs = self.transformer(input_ids=input_ids.long())
        speculative_logits = self.speculative_head(outputs.last_hidden_state)
        
        # Reshape logits for multi-token loss calculation
        speculative_logits = speculative_logits.view(
            speculative_logits.size(0), 
            -1, 
            self.vocab_size
        )
        
        # Compute loss across multiple tokens
        loss = self._compute_multi_token_loss(
            speculative_logits, 
            target, 
            attention_mask
        )
        
        data_dict['loss'] = self.loss_wrapper(loss)
        return data_dict

    def _prepare_multi_token_targets(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Prepare multi-token targets for speculative prediction.
        
        Args:
            input_ids (Tensor): Input token IDs
            attention_mask (Tensor): Attention mask
        
        Returns:
            Tensor: Prepared target tensor for multi-token prediction
        """
        target = input_ids.clone()
        target[attention_mask == 0] = -100
        
        # Shift targets to create multi-token prediction scenario
        target_shifts = [target[:, i:] for i in range(1, self.speculative_tokens + 1)]
        return torch.stack(target_shifts, dim=1)

    def _compute_multi_token_loss(
        self, 
        logits: Tensor, 
        targets: Tensor, 
        attention_mask: Tensor
    ) -> Tensor:
        """
        Compute loss for multi-token speculative prediction.
        
        Args:
            logits (Tensor): Predicted logits
            targets (Tensor): Target tokens
            attention_mask (Tensor): Attention mask
        
        Returns:
            Tensor: Computed loss
        """
        # Flatten logits and targets for loss computation
        flattened_logits = logits.view(-1, self.vocab_size)
        flattened_targets = targets.view(-1)
        
        # Create a mask to ignore padded tokens
        mask = (flattened_targets != -100)
        
        # Compute cross-entropy loss
        loss = nnf.cross_entropy(
            flattened_logits[mask], 
            flattened_targets[mask], 
            reduction='mean'
        )
        
        return loss

    @torch.no_grad()
    def speculative_generate(
        self, 
        data_dict: dict = None, 
        num_return_sequences: int = 8, 
        generation_config: Dict = None
    ) -> dict:
        """
        Speculative generation method with parallel token prediction.
        
        Args:
            data_dict (dict): Input data dictionary
            num_return_sequences (int): Number of sequences to generate
            generation_config (Dict): Generation configuration
        
        Returns:
            dict: Generated output
        """
        generation_config = generation_config or {}
        device = next(self.parameters()).device
        max_length = generation_config.get('max_length', 8192)
        
        # Initial input preparation
        input_ids = data_dict.get('input_ids', 
            torch.full((num_return_sequences, 1), self.bos_token_id, device=device)
        )
        
        while input_ids.size(1) < max_length:
            # Compute speculative predictions
            speculative_outputs = self._compute_speculative_tokens(
                input_ids, 
                generation_config
            )
            
            # Append speculative tokens
            input_ids = torch.cat([input_ids, speculative_outputs], dim=1)
            
            # Early stopping conditions
            if self._should_stop_generation(input_ids):
                break
        
        # Post-processing
        output_ids = self._postprocess_generated_ids(input_ids)
        
        return self.tokenizer.detokenize(input_ids=output_ids)

    def _compute_speculative_tokens(
        self, 
        input_ids: Tensor, 
        generation_config: Dict
    ) -> Tensor:
        """
        Compute speculative tokens using the model's parallel prediction mechanism.
        
        Args:
            input_ids (Tensor): Current input sequence
            generation_config (Dict): Generation configuration
        
        Returns:
            Tensor: Speculative tokens
        """
        # Compute transformer outputs
        transformer_outputs = self.transformer(input_ids)
        
        # Use speculative head for multi-token prediction
        speculative_logits = self.speculative_head(
            transformer_outputs.last_hidden_state[:, -1:]
        )
        
        # Reshape and apply temperature
        speculative_logits = speculative_logits.view(
            input_ids.size(0), 
            self.speculative_tokens, 
            -1
        ) / self.speculative_temperature
        
        # Sample tokens with top-k and top-p filtering
        speculative_tokens = self._sample_tokens(speculative_logits, generation_config)
        
        return speculative_tokens

    def _sample_tokens(
        self, 
        logits: Tensor, 
        generation_config: Dict
    ) -> Tensor:
        """
        Sample tokens using top-k and top-p filtering.
        
        Args:
            logits (Tensor): Input logits
            generation_config (Dict): Generation configuration
        
        Returns:
            Tensor: Sampled tokens
        """
        top_k = generation_config.get('top_k', 50)
        top_p = generation_config.get('top_p', 0.95)
        
        # Apply top-k and top-p filtering
        filtered_logits = self._filter_logits(logits, top_k, top_p)
        
        # Sample tokens
        token_probs = torch.softmax(filtered_logits, dim=-1)
        sampled_tokens = torch.multinomial(token_probs.view(-1, token_probs.size(-1)), 1)
        
        return sampled_tokens.view(logits.size(0), -1)

    def _filter_logits(
        self, 
        logits: Tensor, 
        top_k: int, 
        top_p: float
    ) -> Tensor:
        """
        Apply top-k and top-p (nucleus) sampling filtering to logits.
        
        This method reduces the risk of sampling low-probability tokens while 
        maintaining diversity in generation.
        
        Args:
            logits (Tensor): Input logits from the model
            top_k (int): Number of highest probability tokens to keep
            top_p (float): Cumulative probability threshold for nucleus sampling
        
        Returns:
            Tensor: Filtered logits with low-probability tokens masked
        """
        # Create a copy of logits to avoid modifying the original tensor
        filtered_logits = logits.clone()
        
        # Top-K Filtering
        # 1. Limit to top k most probable tokens
        if top_k > 0:
            # For each sequence in the batch
            for i in range(filtered_logits.size(0)):
                # Find the top k values
                top_k_values, top_k_indices = torch.topk(filtered_logits[i], top_k)
                
                # Create a mask of the same shape as logits
                mask = torch.full_like(filtered_logits[i], float('-inf'))
                mask.scatter_(0, top_k_indices, top_k_values)
                
                # Replace logits with the masked version
                filtered_logits[i] = mask
        
        # Top-P (Nucleus) Sampling
        # 2. Reduce tokens based on cumulative probability
        if top_p < 1.0:
            for i in range(filtered_logits.size(0)):
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(filtered_logits[i], descending=True)
                
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
                
                # Create a mask to remove tokens beyond the cumulative probability threshold
                # Tokens that exceed top_p cumulative probability are set to a very low value
                nucleus_mask = cumulative_probs > top_p
                
                # Ensure at least one token remains
                if nucleus_mask.all():
                    nucleus_mask[0] = False
                
                # Scatter the mask back to the original logits
                filtered_logits[i].scatter_(0, sorted_indices, 
                    torch.where(nucleus_mask, 
                        torch.full_like(sorted_logits, float('-inf')), 
                        sorted_logits
                    )
                )
        
        return filtered_logits

    def _should_stop_generation(self, input_ids: Tensor) -> bool:
        """
        Determine if generation should stop.
        
        Args:
            input_ids (Tensor): Current input sequence
        
        Returns:
            bool: Whether to stop generation
        """
        return (
            input_ids.size(1) >= 8192 or  # Max length
            torch.any(input_ids[:, -1] == self.eos_token_id)  # EOS token detected
        )

    def _postprocess_generated_ids(self, input_ids: Tensor) -> Tensor:
        """
        Post-process generated token IDs.
        
        Args:
            input_ids (Tensor): Generated input IDs
        
        Returns:
            Tensor: Processed token IDs
        """
        # Remove BOS token and truncate
        output_ids = input_ids[:, 1:]
        output_ids[output_ids == self.eos_token_id] = self.tokenizer.pad_id
        return output_ids[:, :8191]  # Ensure max length