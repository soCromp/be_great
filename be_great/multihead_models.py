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
        print('generate with mlp', self.col.value)
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
            self.col.value = 0
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
        
        
        def set_train_mode(self):
            self.forward = self.multicol_forward
            
            
        def set_generation_mode(self):
            self.forward = self.autocol_forward
            
        
        def autocol_forward(self, input_ids = None, attention_mask = None, labels = None, **kwargs):
            PAD = 50256
            EOS = 50258
            transformer, lm_head = self.children()
            
            prompt = deepcopy(input_ids) #bs x tokens
            mask = torch.ones_like(prompt)
            
            transformer_outputs = transformer(prompt, attention_mask=mask, **kwargs)
            hidden_states = transformer_outputs[0]
            lm_logits = lm_head(hidden_states)
                    
            # print(lm_logits[:, -1].argmax().item())
            if lm_logits[:, -1].argmax().item() == EOS and self.col.value < self.num_experts-1:
                # print('to next col')
                self.col.value +=1
            
            
            return CausalLMOutputWithPast(
                loss = None,
                logits = lm_logits,
                past_key_values = transformer_outputs.past_key_values,
                hidden_states = transformer_outputs.hidden_states,
                attentions = transformer_outputs.attentions
            )
            
        
        def multicol_forward(self, input_ids = None, attention_mask = None, labels = None, cols_iterator=None, **kwargs):
            # input_ids, attention_mask, labels: batch x column x tokens
            # print('labels', labels)
            PAD = 50256
            EOS = 50258
            transformer, lm_head = self.children()
            
            prompt = deepcopy(input_ids) #bs x tokens
            if labels is not None:
                prompt = torch.cat([prompt[prompt!=PAD].unsqueeze(0), 
                                    labels[:,0,:][labels[:,0,:] != PAD].unsqueeze(0)], axis=1)
            
            if cols_iterator == None:
                cols_iterator = range(self.num_experts)
            elif len(cols_iterator.shape) == 2: # because huggingface trainor wraps cols_iterator into extra []
                cols_iterator = cols_iterator[0]
            
            collosses = []
            lossavg = None
            # print(0, prompt)
            mask = torch.ones_like(prompt)
            for i in cols_iterator:
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
                    
                lossavg = sum(collosses) / len(collosses)
                    
                # update prompt and mask
                if i < self.num_experts-1:
                    if labels is not None: #in training mode, where labels are known
                        prompt = torch.cat([prompt, labels[:,i+1,:][labels[:,i+1,:] != PAD].unsqueeze(0)], axis=1)
                    else: # in inference mode, where a column's prompt is the preds from the prior columns
                        predtoks = lm_logits.argmax(-1)
                        wheredone = torch.where(predtoks == EOS)[-1] # places it predicts the EOS token
                        if len(wheredone) == 0: #didn't find EOS in the predicted tokens
                            eostoks = torch.full((prompt.shape[0],1), EOS) # add EOS at end of col since model didn't so itself
                            prompt = torch.cat([prompt, predtoks, eostoks])
                        else:
                            doneind = wheredone[0].item() # first place it predicts to be done
                            prompt = torch.cat([prompt, predtoks[:, :doneind+1]])
                        
                    mask = torch.ones_like(prompt)
                    
            return CausalLMOutputWithPast(
                loss = lossavg,
                logits = lm_logits,
                past_key_values = transformer_outputs.past_key_values,
                hidden_states = transformer_outputs.hidden_states,
                attentions = transformer_outputs.attentions
            )
    
     
    return MOEModelForCausalLM.from_other(model, **kwargs)
