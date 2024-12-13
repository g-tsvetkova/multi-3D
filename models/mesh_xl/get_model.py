import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from transformers import OPTConfig, OPTForCausalLM
from models.mesh_xl.tokenizer import MeshTokenizer
from typing import Dict


class MeshXL(nn.Module):
    
    def train(self, mode: bool = True):
        super().train(mode)
        return self
    
    def __init__(self, args):
        super().__init__()
        
        self.tokenizer = MeshTokenizer(args)
        
        # causal LM model initialization
        self.vocab_size = self.tokenizer.codebook_size + 3
        self.bos_token_id = self.tokenizer.codebook_size
        self.eos_token_id = self.tokenizer.codebook_size + 1
        self.pad_token_id = self.tokenizer.codebook_size + 2
        
        # config = AutoConfig.from_pretrained(
        #     args.llm, 
        #     n_positions=8192,
        #     max_position_embeddings=8192,
        #     vocab_size=self.vocab_size,
        #     bos_token_id=self.bos_token_id,
        #     eos_token_id=self.eos_token_id,
        #     pad_token_id=self.pad_token_id
        # )

        # Create a custom OPT configuration
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
            ffn_dim = 1024
        )
        print(self.vocab_size)

        config.word_embed_proj_dim = config.hidden_size
        # self.transformer = AutoModelForCausalLM.from_config(config)
        self.transformer = OPTForCausalLM(config)
        # self._init_weights(self.transformer)
        
        # setting status for all parameters
        self.train()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1000)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1000)
            if hasattr(module, 'padding_idx'):
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
            self, 
            data_dict: dict=None, 
            is_eval: bool=False, 
            is_generate: bool=False,
            num_return_sequences: int=8, 
            generation_config: Dict=dict(
                do_sample=True,
                top_k=50,
                top_p=0.95,
                # no_repeat_ngram_size=9,
            )
        ) -> dict:
        
        if not is_eval:
            return self.train_one_step(data_dict)
        
        if is_eval and not is_generate:
            return self.perplexity(data_dict)
        
        if is_eval and is_generate:
            return self.generate(
                data_dict=data_dict, 
                num_return_sequences=num_return_sequences, 
                generation_config=generation_config
            )
        
        raise NotImplementedError('training status undefined!')
        return 

    def loss_wrapper(self, loss: Tensor) -> Tensor:
        # parameter activation: it is a l2 loss with 0 weight
        for param in self.parameters():
            loss += 0 * torch.sum(param ** 2)
        return loss
    
    def train_one_step(self, data_dict: dict) -> dict:
        
        data_dict = self.tokenizer.tokenize(data_dict)
        
        input_ids = data_dict['input_ids']              # batch x ntoken
        attention_mask = data_dict['attention_mask']    # batch x ntoken
        
        # parse input with <bos> and <eos> tokens
        input_ids[input_ids == self.tokenizer.pad_id] = self.pad_token_id   # <pad> xxx <pad> <pad>
        input_ids[:, 0] = self.bos_token_id                                 # <bos> xxx <pad> <pad>
        eos_pos_id = attention_mask.sum(1, keepdim=True) - 1
        input_ids = torch.scatter(                                          # <bos> xxx <eos> <pad>
            input_ids, 
            1, 
            eos_pos_id.long(), 
            torch.ones_like(input_ids) * self.eos_token_id
        )
        
        target = input_ids.clone()
        target[attention_mask == 0] = -100              # not loss for the padding tokens
        
        # Forward pass, calling causal llm
        output = self.transformer(
            input_ids = input_ids.long(),
        )
        
        # compute loss with shift one-token right
        logit = output.logits[:, :-1]   # batch x ntoken x vocab
        label = target[:, 1:]           # batch x ntoken
        
        final_loss = nnf.cross_entropy(
            logit.permute(0, 2, 1),         # batch x vocab x ntoken
            label,
        )   # batch x ntoken
        
        data_dict['loss'] = self.loss_wrapper(final_loss)
        data_dict['gen_loss'] = final_loss
        
        # # Print loss and some weights for verification
        # if torch.distributed.get_rank() == 0:
        #     print(f"Loss: {final_loss.item()}")
        #     print(f"First 5 weights of embed_tokens: {self.transformer.model.decoder.embed_tokens.weight.data[:5, :5]}")
        
        return data_dict
    
    @torch.no_grad()
    def perplexity(self, data_dict: dict) -> dict:
        
        data_dict = self.tokenizer.tokenize(data_dict)
        
        input_ids = data_dict['input_ids']              # batch x ntoken
        attention_mask = data_dict['attention_mask']    # batch x ntoken
        
        # set pad_token_id = eos_token_id
        input_ids[input_ids == self.tokenizer.pad_id] = self.pad_token_id   # <pad> xxx <pad> <pad>
        input_ids[:, 0] = self.bos_token_id                                 # <sos> xxx <pad> <pad>
        eos_pos_id = attention_mask.sum(1, keepdim=True) - 1
        input_ids = torch.scatter(                                          # <bos> xxx <eos> <pad>
            input_ids, 
            1, 
            eos_pos_id.long(), 
            torch.ones_like(input_ids) * self.eos_token_id
        )
        
        # llm loss calculation
        output = self.transformer(
            input_ids = input_ids.long(),
        )
        
        # compute loss with shift token right
        logit = output.logits[:, :-1]   # batch x (ntoken - 1) x vocab
        label = input_ids[:, 1:]        # batch x (ntoken - 1)
        masks = attention_mask[:, 1:]   # batch x (ntoken - 1)
        loss_per_token = nnf.cross_entropy(
            logit.permute(0, 2, 1),     # batch x (ntoken - 1) x ntoken
            label,                      # batch x (ntoken - 1)
            reduction='none'
        )   # batch x ntoken
        
        # compute negative log likelihood for each sequence
        neg_log_likelihood = torch.sum(loss_per_token * masks, dim=1) / torch.sum(masks, dim=1)
        
        data_dict['neg_log_likelihood'] = neg_log_likelihood    # batch,
        return data_dict
    
    @torch.no_grad()
    def generate(self, data_dict: dict=None, num_return_sequences: int=8, generation_config: dict=dict()) -> dict:

        net_device = next(self.parameters()).device
        max_length = 8192
        output_ids = torch.ones(num_return_sequences, max_length).long().to(net_device) * self.eos_token_id
        
        # batch x ntokens
        results = self.transformer.generate(
            max_new_tokens=max_length-1,
            num_return_sequences=num_return_sequences,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
            **generation_config
        )
        output_ids[:, :results.shape[1]] = results
        
        # discard <bos> and <eos> tokens to pad tokens
        output_ids = output_ids[:, 1: -1]
        output_ids[output_ids == self.eos_token_id] = self.tokenizer.pad_id
        
        decoder_output = self.tokenizer.detokenize(input_ids=output_ids)
        
        return decoder_output
    
    

def get_model(args):
    model = MeshXL(args)
    return model

