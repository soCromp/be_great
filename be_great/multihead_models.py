from copy import deepcopy
from collections import OrderedDict
import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

#generation
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.utils import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput

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
        # print('generate with mlp', self.col.value)
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
            
            
        def set_generation_mode(self, token_heads=None, column_names_tokens=None):
            self.forward = self.autocol_forward
            if token_heads==None:
                self.token_heads = list(range(self.num_experts))
            else:
                self.token_heads = token_heads
            self.column_names_tokens = column_names_tokens
            
        
        #generation forward
        def autocol_forward(self, input_ids = None, attention_mask = None, labels = None, **kwargs):
            EOS = 50258
            transformer, lm_head = self.children()
            
            prompt = deepcopy(input_ids) #bs x tokens
            mask = torch.ones_like(prompt)
            
            # print('mlps col', self.col.value)
            transformer_outputs = transformer(prompt, attention_mask=mask, **kwargs)
            hidden_states = transformer_outputs[0]
            lm_logits = lm_head(hidden_states)
                    
            # print(lm_logits[:, -1].argmax().item())
            # if lm_logits[:, -1].argmax().item() == EOS and self.col.value < self.num_experts-1:
            #     # print('to next col')
            #     self.col.value +=1
            
            
            return CausalLMOutputWithPast(
                loss = None,
                logits = lm_logits,
                past_key_values = transformer_outputs.past_key_values,
                hidden_states = transformer_outputs.hidden_states,
                attentions = transformer_outputs.attentions
            )
            
            
        #training forward
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
            # print(cols_iterator)
            
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
    
    
        def _sample(
            self,
            input_ids,
            logits_processor = None,
            stopping_criteria = None,
            logits_warper = None,
            max_length = None,
            pad_token_id = None,
            eos_token_id = None,
            output_attentions = None,
            output_hidden_states = None,
            output_scores = None,
            output_logits = None,
            return_dict_in_generate = None,
            synced_gpus = False,
            streamer = None,
            **model_kwargs,
        ):
            def get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper):
                EOS = 50258
                # if input_ids[..., -1] == EOS:
                #     next_token_scores = torch.full_like(next_token_logits, -1*float("Inf"))
                #     next_token_scores[..., ?] = float("Inf")
                next_token_scores = logits_processor(input_ids, next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)
                return next_token_scores
            def select_next_token(next_token_scores):
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                return next_tokens
            
            return self.generation_loop(input_ids, logits_processor, stopping_criteria, logits_warper, max_length, pad_token_id, eos_token_id, output_attentions, 
                output_hidden_states, output_scores, output_logits, return_dict_in_generate, synced_gpus, streamer,
                get_next_token_scores, select_next_token, **model_kwargs)
        
        
        def _greedy_search(
            self,
            input_ids,
            logits_processor = None,
            stopping_criteria = None,
            max_length = None,
            pad_token_id = None,
            eos_token_id = None,
            output_attentions = None,
            output_hidden_states = None,
            output_scores = None,
            output_logits = None,
            return_dict_in_generate = None,
            synced_gpus = False,
            streamer = None,
            **model_kwargs,
        ):
            def get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper=None):
                next_token_scores = logits_processor(input_ids, next_token_logits)
                return next_token_scores
            def select_next_token(next_token_scores):
                next_tokens = torch.argmax(next_token_scores, dim=-1)
                return next_tokens
                
            return self.generation_loop(input_ids, logits_processor, stopping_criteria, None, max_length, pad_token_id, eos_token_id, output_attentions, 
                output_hidden_states, output_scores, output_logits, return_dict_in_generate, synced_gpus, streamer,
                get_next_token_scores, select_next_token, **model_kwargs)
    
    
        def generation_loop(
            self,
            input_ids,
            logits_processor,
            stopping_criteria,
            logits_warper,
            max_length,
            pad_token_id,
            eos_token_id,
            output_attentions,
            output_hidden_states,
            output_scores,
            output_logits,
            return_dict_in_generate,
            synced_gpus,
            streamer,
            get_next_token_scores, 
            select_next_token,
            **model_kwargs,
        ):
            # init values
            EOS = 50258
            expert = 1 # used to index into the list saying the order of cols/experts
            self.col.value = self.token_heads[expert]
            # print(self.col.value)
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
            if max_length is not None:
                warnings.warn(
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                    UserWarning,
                )
                stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
            logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
            pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
            eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
            output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
            output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
            output_attentions = (
                output_attentions if output_attentions is not None else self.generation_config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            )
            return_dict_in_generate = (
                return_dict_in_generate
                if return_dict_in_generate is not None
                else self.generation_config.return_dict_in_generate
            )

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            if "inputs_embeds" in model_kwargs:
                cur_len = model_kwargs["inputs_embeds"].shape[1]
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
            
            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # forward pass to get next token
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_token_scores = get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # choose next tokens (sample/argmax)
                next_tokens = select_next_token(next_token_scores)
                if input_ids[..., -1].item() == EOS and expert < self.num_experts-1:
                    next_tokens = torch.full_like(next_tokens, self.column_names_tokens[expert][0])
                    expert += 1
                    self.col.value = self.token_heads[expert]
                    
                # if next_tokens.item() == EOS and expert < self.num_experts-1:
                #     expert += 1
                #     self.col.value = self.token_heads[expert]
                #     for token in self.column_names_tokens[expert]:
                #         print(token)
                #         model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                #         outputs = self(
                #             **model_inputs,
                #             return_dict=True,
                #             output_attentions=output_attentions,
                #             output_hidden_states=output_hidden_states,
                #         )
                #         next_token_logits = outputs.logits[:, -1, :]
                #         # pre-process distribution
                #         # next_token_scores = get_next_token_scores(input_ids, next_token_logits, logits_processor, logits_warper)
                #         next_tokens = torch.full_like(next_tokens, token)
                #         print('before', input_ids.shape)
                #         input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                #         print('after', input_ids.shape)
                #         model_kwargs = self._update_model_kwargs_for_generation(
                #             outputs,
                #             model_kwargs,
                #             is_encoder_decoder=self.config.is_encoder_decoder,
                #         )

                # finished sentences should have their next token be a padding token
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences.mul(
                        next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )

                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0

            if streamer is not None:
                streamer.end()

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            else:
                return input_ids
     
     
    return MOEModelForCausalLM.from_other(model, **kwargs)
