from copy import deepcopy
from collections import OrderedDict
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

SUPERCLASS_FOR_HEADLESS_LM = {GPT2LMHeadModel:GPT2Model, LlamaForCausalLM:LlamaModel}

class Integer: # so I can have pointers to an int
    value = 0

class MOEMLP(nn.Module):
    def __init__(self, col):
        super(MOEMLP, self).__init__()
        self.col = col
        
    def from_other(mlp, col):
        moemlp = MOEMLP(col)
        mlpslist = []
        for i in range(num_experts):
            mlpslist.append(deepcopy(mlp))
        moemlp.mlps = nn.ModuleList(mlpslist)
        return moemlp
        
        
    def forward(self, hidden_states):
        print(self.col.value)
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
        
    
def MOEModelForCausalLM(model):
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
            self.num_experts=3
            
        def from_other(model):
            # https://stackoverflow.com/questions/597199/converting-an-object-into-a-subclass-in-python
            moemodel = deepcopy(model)
            moemodel.__class__ = MOEModelForCausalLM
            moemodel.col = Integer()
            
            if type(model) == GPT2LMHeadModel:
                for i in range(len(moemodel.transformer.h)):
                    moemodel.transformer.h[i].mlp = MOEMLP.from_other(moemodel.transformer.h[i].mlp, moemodel.col)
                    # moemodel.moemlps.append(moemodel.transformer.h[i].mlp)
            elif type(model) == LlamaForCausalLM:
                moemodel = deepcopy(model)
                print('deep copied model')
                moemodel.__class__ = MOEModelForCausalLM
                for i in range(len(moemodel.model.layers)):
                    moemodel.model.layers[i].mlp = MOEMLP.from_other(moemodel.model.layers[i].mlp, moemodel.col)
                    # moemodel.moemlps.append(moemodel.model.layers[i].mlp)
                print('added MOE MLPs')
            else:
                raise NotImplementedError(f'Type {type(model)} not supported')
            return moemodel
        
        def forward(self, input_ids = None, attention_mask = None, labels = None, **kwargs):
            transformer, lm_head = self.children()
            
            loss = None
            for i in range(self.num_experts):
                self.col.value = i
                transformer_outputs = transformer(input_ids, attention_mask=attention_mask, **kwargs)
                hidden_states = transformer_outputs[0]
                lm_logits = lm_head(hidden_states)

            
            if labels is not None:
                print('calc loss')
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
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
     
    return MOEModelForCausalLM.from_other(model)
