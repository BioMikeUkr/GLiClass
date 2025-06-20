from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import LayerNorm, DebertaV2Layer, build_relative_position
import torch
import torch.nn as nn
import copy
from abc import abstractmethod
from typing import List, Optional, Tuple
from .config import CrossEncoderHeadConfig


class BaseCrossEncodingHead(PreTrainedModel):
    """
    Base class for cross-encoding heads.
    """
    def __init__(self):
        super().__init__()

    def construct_premise_hypothesis_inputs(self, token_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        batch_size, seq_len, hidden_dim = token_embeds.shape
        device = token_embeds.device

        class_token_mask = input_ids == self.config.class_token_index
        text_token_mask = input_ids == self.config.text_token_index

        class_batch_idx, class_token_idx = torch.where(class_token_mask)
        _, bos_token_idx = torch.where(text_token_mask)

        reversed_attention = attention_mask.flip(dims=[1])
        last_token_idx = seq_len - reversed_attention.float().argmax(dim=1) - 1
        eos_mask = torch.zeros_like(attention_mask, dtype=torch.bool).scatter(1, last_token_idx.unsqueeze(1), True)
        _, eos_token_idx = torch.where(eos_mask)

        # Build mapping: (batch, class_start, class_end, text_start, text_end)
        all_pairs = []

        for b in range(batch_size):
            cls_indices = class_token_idx[class_batch_idx == b]
            if cls_indices.numel() == 0:
                continue
            cls_positions = torch.cat([cls_indices, bos_token_idx[b].unsqueeze(0)], dim=0)
            cls_start = cls_positions[:-1]
            cls_end = cls_positions[1:]
            text_start = bos_token_idx[b]
            text_end = eos_token_idx[b]
            batch_idx = torch.full_like(cls_start, b)

            batch_pairs = torch.stack([batch_idx, cls_start, cls_end,
                                    torch.full_like(cls_start, text_start),
                                    torch.full_like(cls_start, text_end)], dim=1)
            all_pairs.append(batch_pairs)

        pairs = torch.cat(all_pairs, dim=0)
        n_pairs = pairs.size(0)

        # Compute max sequence length
        cls_lengths = pairs[:, 2] - pairs[:, 1]
        txt_lengths = pairs[:, 4] - pairs[:, 3]
        max_seq_len = (cls_lengths + txt_lengths + 2).max().item()

        cross_embeds = torch.zeros(n_pairs, max_seq_len, hidden_dim, dtype=token_embeds.dtype, device=device)
        cross_ids = torch.zeros(n_pairs, max_seq_len, dtype=input_ids.dtype, device=device)

        for i in range(n_pairs):
            b, cls_start, cls_end, txt_start, txt_end = pairs[i]
            b = int(b)

            cls_range = input_ids[b, cls_start:cls_end]
            txt_range = input_ids[b, txt_start:txt_end]
            cls_embeds = token_embeds[b, cls_start:cls_end]
            txt_embeds = token_embeds[b, txt_start:txt_end]

            bos_id = input_ids[b, 0]
            eos_id = input_ids[b, txt_end]
            bos_embed = token_embeds[b, 0]
            eos_embed = token_embeds[b, txt_end]

            cls_len = cls_range.size(0)
            txt_len = txt_range.size(0)

            cls_start_pos = 1
            txt_start_pos = cls_start_pos + cls_len
            eos_pos = txt_start_pos + txt_len

            cross_embeds[i, 0] = bos_embed
            cross_embeds[i, cls_start_pos:cls_start_pos + cls_len] = cls_embeds
            cross_embeds[i, txt_start_pos:txt_start_pos + txt_len] = txt_embeds
            cross_embeds[i, eos_pos] = eos_embed

            cross_ids[i, 0] = bos_id
            cross_ids[i, cls_start_pos:cls_start_pos + cls_len] = cls_range
            cross_ids[i, txt_start_pos:txt_start_pos + txt_len] = txt_range
            cross_ids[i, eos_pos] = eos_id

        cross_attention_mask = (cross_ids > 0).long()

        return {
            "mapping": pairs,
            "cross_embeddings": cross_embeds,
            "cross_input_ids": cross_ids,
            "cross_attention_mask": cross_attention_mask,
        }

    @abstractmethod
    def forward(self, text_rep: torch.Tensor, label_rep: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")


class DebertaV2CrossEncoderHead(PreTrainedModel):
    config_class = CrossEncoderHeadConfig

    def __init__(self, config: CrossEncoderHeadConfig, encoder_model: Optional[nn.Module] = None):
        super().__init__(config)

        self.active_layers = config.active_layers
        self.z_steps = config.z_steps
        self.inner_batch_size = config.inner_batch_size
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
    
    def forward(
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

        if self.inner_batch_size > len(cross_embeddings):
            
        encoder_outputs = self.encode(
            cross_embeddings,
            cross_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict
        )

        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.LayerNorm(sequence_output[:, 0, :])

        return BaseModelOutput(
            last_hidden_state=pooled_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None
        )
    
# Example dictionary for scorers
CROSSBACKBONE2OBJECT = {
    "deberta-v2": DebertaV2CrossEncoderHead, 
}

теперь мне нужно заставить эту штуку разбивать на батчи по иннер батч сайз и собирать как-то обратно