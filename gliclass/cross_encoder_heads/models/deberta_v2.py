from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import LayerNorm, DebertaV2Layer, build_relative_position
import torch
import torch.nn as nn
import copy
from typing import List, Optional, Tuple
from ...config import GLiClassModelConfig
from ..base import CrossEncoderHeadOutput, BaseCrossEncoderHead


class DebertaV2CrossEncoderHead(BaseCrossEncoderHead):

    def __init__(self, config: GLiClassModelConfig, encoder_model: Optional[nn.Module] = None):
        super().__init__(config)
        encoder_config = config.encoder_config

        if encoder_model is not None:
            self.cross_encoder = nn.ModuleList([
                copy.deepcopy(encoder_model.encoder.layer[i]) for i in self.active_layers
            ])
        else:
            self.cross_encoder = nn.ModuleList([
                DebertaV2Layer(encoder_config) for _ in self.active_layers
            ])

        if encoder_model is not None and hasattr(encoder_model, "LayerNorm"):
            self.LayerNorm = copy.deepcopy(encoder_model.LayerNorm)
        else:
            self.LayerNorm = LayerNorm(
                encoder_config.hidden_size,
                encoder_config.layer_norm_eps,
                elementwise_affine=True
            )

        self.relative_attention = getattr(encoder_config, "relative_attention", False)

        if self.relative_attention:
            if encoder_model is not None and hasattr(encoder_model, "rel_embeddings"):
                self.rel_embeddings = copy.deepcopy(encoder_model.rel_embeddings)
            else:
                self.max_relative_positions = getattr(encoder_config, "max_relative_positions", -1)
                if self.max_relative_positions < 1:
                    self.max_relative_positions = encoder_config.max_position_embeddings

                self.position_buckets = getattr(encoder_config, "position_buckets", -1)
                pos_ebd_size = self.max_relative_positions * 2
                if self.position_buckets > 0:
                    pos_ebd_size = self.position_buckets * 2

                self.rel_embeddings = nn.Embedding(pos_ebd_size, encoder_config.hidden_size)
        else:
            self.rel_embeddings = None

        self.norm_rel_ebd = [
            x.strip() for x in getattr(encoder_config, "norm_rel_ebd", "none").lower().split("|")
        ]

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            if query_states is not None:
                relative_pos = build_relative_position(
                    query_states,
                    hidden_states,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
            else:
                relative_pos = build_relative_position(
                    hidden_states,
                    hidden_states,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
        return relative_pos
    
    def encode(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    ):
        rel_embeddings = self.get_rel_embedding()
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states)

        all_hidden_states: Optional[Tuple[torch.Tensor]] = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        next_kv = hidden_states

        for _ in range(self.z_steps):
            for layer_module in self.cross_encoder:

                output_states, attn_weights = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=None,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )
                
                if output_attentions:
                    all_attentions = all_attentions + (attn_weights,)

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (output_states,)

                next_kv = output_states


        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        
        encoder_outputs =  BaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

        sequence_output = all_hidden_states[-1]
        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]
        
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )
    
    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_embeds: torch.Tensor,
        inner_batch_size: Optional[int] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput:
        if token_embeds is None:
            raise ValueError("token_embeds must be provided for DebertaV2CrossEncoderHead.")

        cross_inputs = self.construct_premise_hypothesis_inputs(token_embeds, input_ids, attention_mask)
        cross_embeddings = cross_inputs["cross_embeddings"]
        cross_attention_mask = cross_inputs["cross_attention_mask"]

        total_size = cross_embeddings.size(0)
        inner_bs = inner_batch_size or self.inner_batch_size or total_size

        last_hidden_list = []
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for start in range(0, total_size, inner_bs):
            end = min(start + inner_bs, total_size)
            chunk_embeddings = cross_embeddings[start:end]
            chunk_attention_mask = cross_attention_mask[start:end]

            encoder_outputs = self.encode(
                chunk_embeddings,
                chunk_attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True,
            )

            last_hidden_list.append(encoder_outputs.last_hidden_state)
            if output_hidden_states and encoder_outputs.hidden_states is not None:
                all_hidden_states.extend(encoder_outputs.hidden_states)
            if output_attentions and encoder_outputs.attentions is not None:
                all_attentions.extend(encoder_outputs.attentions)

        last_hidden_state = torch.cat(last_hidden_list, dim=0)
        # if return_dict:
        #     return self._create_output(
        #         last_hidden_state,
        #         cross_inputs,
        #         all_hidden_states,
        #         all_attentions,
        #         output_hidden_states,
        #         output_attentions,
        #     )
        return CrossEncoderHeadOutput(
            cross_embeddings =last_hidden_state,
            batch_mapping=cross_inputs["batch_mapping"],
            cross_input_ids=cross_inputs["cross_input_ids"],
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_attentions if output_attentions else None,
        )

    
# # Example dictionary for scorers
# CROSSBACKBONE2OBJECT = {
#     "deberta-v2": DebertaV2CrossEncoderHead, 
# }