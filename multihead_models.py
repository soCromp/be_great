from copy import deepcopy
from collections import OrderedDict
import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

SUPERCLASS_FOR_HEADLESS_LM = {GPT2LMHeadModel:GPT2Model, LlamaForCausalLM:LlamaModel}

class Integer: # so I can have pointers to an int
    value = 0

class MOEMLP(nn.Module):
    def __init__(self, col):
        super(MOEMLP, self).__init__()
        self.col = col
        
    def from_other(mlp, col, num_experts):
        moemlp = MOEMLP(col)
        mlpslist = []
        for i in range(num_experts):
            mlpslist.append(deepcopy(mlp))
        moemlp.mlps = nn.ModuleList(mlpslist)
        return moemlp
        
        
    def forward(self, hidden_states):
        # print(self.col.value)
        return self.mlps[self.col.value](hidden_states)


# class MOETransformerBlock(nn.Module):
#     def __init__(self, block):
#         super(MOETransformerBlock, self).__init__()
        
#     def from_other(block):
#         moeblock = deepcopy(block)
#         if type(block) == GPT2Block:
#             moeblock.input_layernorm = block.ln_1
#             moeblock.attention = block.attn
#             moeblock.post_attention_layernorm = block.ln_2
#             moeblock.mlps = MOEMLP(block.mlp)
#         elif type(block) == LlamaDecoderLayer:
#             moeblock.input_layernorm = block.input_layernorm
#             moeblock.attention = block.self_attn
#             moeblock.post_attention_layernorm = block.post_attention_layernorm
#             moeblock.mlps = MOEMLP(block.mlp)
#         else:
#             raise NotImplementedError(f'Type {type(block)} not supported')
#         return moeblock
        
#     def forward(
#         self,
#         hidden_states,
#         attention_mask,
#         col = 0,
#         **kwargs
#     ):
#         residual = hidden_states
#         hidden_states = self.input_layernorm(hidden_states)
        
#         attn_outs = self.attn(
#             hidden_states,
#             attention_mask,
#             **kwargs
#         )
#         hidden_states = attn_outs[0]
#         hidden_states = residual + hidden_states 
        
#         # fully connected 
#         residual = hidden_states 
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states, col)
#         hidden_states = hidden_states + residual 
        
#         outputs = (hidden_states) + attn_outs[1:]
        
    
def MOEModelForCausalLM(model, **kwargs):
    superclassForCausalLM = type(model)
    superclassForHeadlessLM = SUPERCLASS_FOR_HEADLESS_LM[type(model)]
    
    # class MOEModel(superclassForHeadlessLM):
    #     def __init__(self, model):
    #         super(MOEModel, self).__init__()
            
    #     def from_other(model):
    #         moemodel = deepcopy(model)
    #         moemodel.__class__ = MOEModel
    #         moemodel.std_layers = {}
            
    #         if type(model) == GPT2Model:
    #             moemodel.std_layers['pretransformer'] = ['wte', 'wpe', 'drop']
                
    #             moemodel.std_layers['transformer'] = 'h'
    #             moemodel.h = nn.ModuleList([MOETransformerBlock.from_other(block) for block in model.h])

    #             moemodel.std_layers['postnorm'] = 'ln_f'
                
    #         elif type(model) == LlamaModel:
    #             moemodel.std_layers['pretransformer'] = ['embed_layers']
                
    #             moemodel.std_layers['transformer'] = 'layers'
    #             moemodel.layers = nn.ModuleList([MOETransformerBlock.from_other(block) for block in model.layers])
                
    #             moemodel.std_layers['postnorm'] = 'norm'
    #         else:
    #             raise NotImplementedError(f'Type {type(block)} not supported')
            
    
    class MOEModelForCausalLM(superclassForCausalLM):
        def __init__(self):
            super(MOEModelForCausalLM, self).__init__()
            self.col = Integer()
            self.num_experts=1
            
        def from_other(model, num_experts=1):
            # https://stackoverflow.com/questions/597199/converting-an-object-into-a-subclass-in-python
            moemodel = deepcopy(model)
            moemodel.__class__ = MOEModelForCausalLM
            moemodel.col = Integer()
            moemodel.num_experts = num_experts
            
            if type(model) == GPT2LMHeadModel:
                for i in range(len(moemodel.transformer.h)):
                    moemodel.transformer.h[i].mlp = MOEMLP.from_other(
                        moemodel.transformer.h[i].mlp, moemodel.col, moemodel.num_experts)
            elif type(model) == LlamaForCausalLM:
                moemodel = deepcopy(model)
                print('deep copied model')
                moemodel.__class__ = MOEModelForCausalLM
                for i in range(len(moemodel.model.layers)):
                    moemodel.model.layers[i].mlp = MOEMLP.from_other(
                        moemodel.model.layers[i].mlp, moemodel.col, moemodel.num_experts)
                print('added MOE MLPs')
            else:
                raise NotImplementedError(f'Type {type(model)} not supported')
            return moemodel
        
        def forward(self, input_ids = None, attention_mask = None, labels = None, **kwargs):
            # input_ids, attention_mask, labels: batch x column x tokens
            # print('labels', labels)
            PAD = 50256
            transformer, lm_head = self.children()
            
            loss = None
            # print(input_ids, attention_mask, labels)
            
            prompt = deepcopy(input_ids) #bs x tokens
            if labels is not None:
                prompt = torch.cat([prompt[prompt!=PAD], labels[:,0,:][labels[:,0,:] != PAD].unsqueeze(0)], axis=1)
                
            mask = torch.ones_like(prompt)
            
            # print('prompt, mask', prompt, mask)
                
            # for i in range(self.num_experts):
            self.col.value = 0
            transformer_outputs = transformer(prompt, attention_mask=mask, **kwargs)
            hidden_states = transformer_outputs[0]
            lm_logits = lm_head(hidden_states)
            
            # if prompt is None:
            #     prompt = labels[0]
            # else:
            #     prompt = torch.cat([prompt, labels[i]])
            
            # print(labels.shape, lm_logits.shape)

            
            if labels is not None:
                # print('calc loss')
                # move labels to correct device to enable model parallelism
                # labels = labels[:,0,:].to(lm_logits.device)
                # Shift so that tokens < n predict n
                # print('prompt', prompt, prompt.shape, 
                #       'logits', lm_logits.argmax(axis=-1), lm_logits.shape)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = prompt[..., 1:].contiguous()
                # print('shift_labels', shift_labels, shift_labels.shape, 
                #       'shift_logits', shift_logits.argmax(axis=-1), shift_logits.shape)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return CausalLMOutputWithPast(
                loss = loss,
                logits = lm_logits,
                past_key_values = transformer_outputs.past_key_values,
                hidden_states = transformer_outputs.hidden_states,
                attentions = transformer_outputs.attentions
            )
            
        
        def multicol_forward(self, input_ids = None, attention_mask = None, labels = None, **kwargs):
            # input_ids, attention_mask, labels: batch x column x tokens
            # print('labels', labels)
            PAD = 50256
            transformer, lm_head = self.children()
            
            loss = None
            
            prompt = deepcopy(input_ids) #bs x tokens
            if labels is not None:
                prompt = torch.cat([prompt[prompt!=PAD], labels[:,0,:][labels[:,0,:] != PAD].unsqueeze(0)], axis=1)
                
            mask = torch.ones_like(prompt)
            for i in range(self.num_experts):
                self.col.value = i 
                transformer_outputs = transformer(prompt, attention_mask=mask, **kwargs)
                hidden_states = transformer_outputs[0]
                lm_logits = lm_head(hidden_states)
                
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = prompt[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    collosses.append(loss)
                    
                # update prompt and mask
                if i < self.num_experts-1:
                    prompt = torch.cat([prompt, labels[:,i+1,:][labels[:,i+1,:] != PAD].unsqueeze(0)], axis=1)
                    mask = torch.ones_like(prompt)

            return CausalLMOutputWithPast(
                loss = loss,
                logits = lm_logits,
                past_key_values = transformer_outputs.past_key_values,
                hidden_states = transformer_outputs.hidden_states,
                attentions = transformer_outputs.attentions
            )
     
    return MOEModelForCausalLM.from_other(model, **kwargs)
