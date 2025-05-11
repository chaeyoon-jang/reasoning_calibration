# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modified by Chaeyun Jang (https://github.com/chaeyoon-jang).
# This source code is licensed under the terms of the MIT license.
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn 
from collections import namedtuple
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import DynamicCache 

MAX_N_LATENT = 8 

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

class Coconut(nn.Module):
    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_token_id,
        end_latent_token_id,
        eos_token_id,
        target_hidden_idx=-1,
    ):
        super(Coconut, self).__init__()
        
        self.gen_forward_cnt = 0 ## Count for the number of forward passes
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.start_token_id = start_token_id
        self.end_latent_token_id = end_latent_token_id
        self.eos_token_id = eos_token_id
        self.target_hidden_idx = target_hidden_idx
        self.embedding = self.base_causallm.get_input_embeddings()
        
    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None,
                position_ids=None,
                **kwargs):
        """
        Forward pass for the Cocounut model:
        
        Args:
        input_ids: (batch_size, max_seq_length)
        e.g., tensor([[xxxx<latent>xx--],[--xx<latent><latent>xxx]])
        attention_mask: (batch_size, max_seq_length)
        e.g., tensor([[1,1,1,1,1,1,1,0,0], [0,0,1,1,1,1,1,1,1]])
        labels: (batch_size, max_seq_length)
        e.g., tensor([[-100,-100,-100,-100,-100,x,x,-100,-100],
                      [-100,-100,-100,-100,-100,-100,x,x,x]])
        position_ids: (batch_size, max_seq_length)
        e.g., tensor([[0,1,2,3,4,5,6,0,0],[0,0,0,1,2,3,4,5,6]])
        
        Returns:
            loss: torch.Tensor, the loss value for the current forward pass.
            inputs_embeds: torch.Tensor, the input embeddings 
                          for the current forward pass.
            logits: torch.Tensor, the logits for the current forward pass.
        """
        
        batch_size, max_seq_length = input_ids.shape
        latent_indices = (input_ids == self.latent_token_id).nonzero() 
        # e.g., tensor([[0, 4], [1, 4], [1, 5]])
        latent_lists = [[] for _ in range(batch_size)]
        for batch_idx, token_idx in latent_indices:
            latent_lists[batch_idx].append(token_idx.item())
        # e.g., [[4], [4, 5]]
        
        max_n_latents = max(len(latent_list) for latent_list in latent_lists)
        # e.g., 2
        next_compute_range = (
            0, 
            max_seq_length
            if max_n_latents == 0
            else latent_indices[:, 1].min().item()
        ) # e.g., (0, 4)
        
        inputs_embeds = self.embedding(input_ids) 
        # (batch_size, max_seq_length, hidden_size): (2, 9, hidden_size)
        
        logits, kv_cache = [], None 
        
        for pass_idx in range(max_n_latents):
            past_key_values = DynamicCache() if kv_cache is None else kv_cache
            #past_key_values = None if kv_cache is None else [
            #    (
            #        k[:, :, :next_compute_range[0], :],
            #        v[:, :, :next_compute_range[0], :]
            #    )
            #    for k, v in kv_cache
            #]
            outputs = self.base_causallm(
                inputs_embeds = inputs_embeds[
                    :, next_compute_range[0] : next_compute_range[1], :
                ],
                attention_mask = attention_mask[
                    :, next_compute_range[0] : next_compute_range[1]
                ],
                position_ids = position_ids[
                    :, next_compute_range[0] : next_compute_range[1]
                ],
                past_key_values = past_key_values,
                use_cache = True,
                output_hidden_states = True, #TODO: use_cache True
            ) # e.g., pass_idx, next_compute_range: input_ids
            #       0, (0, 4): tensor([[xxxx],[--xx]]) 
            #       1, (4, 5): tensor([[****<latent>],[****<latent>]])
   
            hidden_states_offset = 0 if kv_cache is None \
                else next_compute_range[0]
            
            logits.append(outputs.logits) 
            # (batch_size, max_seq_length, vocab_size): (2, 9, vocab_size)
            
            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1]
                if pass_idx + 1 >= max_n_latents
                else next_compute_range[1] + 1
            ) 
            
            ## Adding latent token
            hidden_states = outputs.hidden_states[self.target_hidden_idx]
            # (batch_size, max_seq_length, hidden_size): (2, 9, hidden_size)
            
            kv_cache = outputs.past_key_values
            
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if pass_idx < len(mask_list)
            ] # e.g., pass_idx, filling_indices
              #       0, [(0, 4), (1, 4)]
              #       1, [(1, 5)]
            
            if filling_indices:
                
                batch_idx_list, token_idx_list = zip(*filling_indices)
                batch_idx_tensor = torch.tensor(batch_idx_list) 
                # e.g., tensor([0, 1]); tensor([1]) 
                token_idx_tensor = torch.tensor(token_idx_list) 
                # e.g., tensor([4, 4]); tensor([5])
                
                # If kv_cache (0, next_compute_range[0]), 
                # then hidden_states start from next_compute_range[0]: 
                # - hidden_states_offset
                # <latent> token is last_hidden_state of previous token: -1
                replace_idx = token_idx_tensor - 1 - hidden_states_offset
                
                inputs_embeds = inputs_embeds.clone()
                
                inputs_embeds[batch_idx_tensor, token_idx_tensor, :] = \
                    hidden_states[batch_idx_tensor, replace_idx, :]
                    
        ## Final pass
        # next_compute_range -> (5, 9): tensor([[xx--],[<latent>xxx]])
        outputs = self.base_causallm(
            inputs_embeds = inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask = attention_mask[
                :, : next_compute_range[1]
            ],
            position_ids=position_ids[
                :, next_compute_range[0] : next_compute_range[1]
            ],
            past_key_values=(
                #[
                #    (
                #        k[:, :, :next_compute_range[0], :], 
                #        v[:, :, :next_compute_range[0], :]
                #    )
                #    for k, v in kv_cache
                #] 
                kv_cache
                if kv_cache 
                else None
            ),
            output_hidden_states=True,
        )
        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1 ## why this is needed?

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CrossEntropyLoss()(shift_logits.view(-1, 
                                                    shift_logits.size(-1)), 
                                  shift_labels.view(-1))
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)
    
    def train(self):
        self.base_causallm.train()
    
    def eval(self):
        self.base_causallm.eval()
    
    def generate(self,
                 input_ids,
                 position_ids,
                 attention_mask=None,
                 max_new_tokens=16,
                 output_embedding=False,
                 synced_gpus=False,
                 **kwargs):
        
        self.gen_forward_cnt = 0
        batch_size = input_ids.shape[0]
        
        ## input_ids: (batch_size, max_seq_length)
        ## e.g., tensor([[xxxx<latent>],[--xx<latent>]])
        
        tokens = [input_ids[i].detach().cpu().tolist() for i in range(batch_size)]
        
        labels = input_ids.clone() # placeholder. not used
                 
        outputs = self.forward(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            labels = labels,
            position_ids = position_ids,
        )
        inputs_embeds = outputs.inputs_embeds 
        ## (batch_size, max_seq_length, hidden_size)
        
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1) 
        ## e.g., tensor([x, x])
        for i in range(batch_size):
            tokens[i].append(next_tokens[i].item())
        
        new_token_embeds = self.embedding(
            torch.tensor(next_tokens, device=input_ids.device),
        ) # (batch_size, embed_dim)
        new_token_embeds = new_token_embeds.unsqueeze(1) 
        # (batch_size, 1, embed_dim)
        new_inputs_embeds = torch.cat(
            (inputs_embeds, new_token_embeds), dim=1
        )
        finished = torch.zeros(batch_size, 
                               dtype=torch.bool, 
                               device=input_ids.device)

        for _ in range(max_new_tokens - 1):
           
            outputs = self.base_causallm(
                inputs_embeds=new_inputs_embeds
            )
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            
            finished |= (next_token == self.eos_token_id)
            
            if finished.all():
                break
            
            for i in range(batch_size):
                if not finished[i]:
                    tokens[i].append(next_token[i].item())
                    
            new_token_embeds = self.embedding(next_token).unsqueeze(1)
            new_inputs_embeds = torch.cat(
                (new_inputs_embeds,
                 new_token_embeds), dim=1
            )
        
        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)
        
        max_len = max(len(t) for t in tokens)
        padded_tokens = []
        for t in tokens:
            padded = t + [self.base_causallm.config.pad_token_id] * (max_len - len(t))
            padded_tokens.append(padded)
        
        output = torch.tensor(padded_tokens, device=input_ids.device)
        
        if output_embedding:
            return output, new_inputs_embeds
        else:
            return output