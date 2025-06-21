from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
import torch
from typing import Tuple
from abc import abstractmethod
from dataclasses import dataclass
from .scorers import MLPScorer, ScorerDot
from .config import CrossEncoderHeadConfig
from ..config import CrossEncoderHeadConfig, GLiClassModelConfig
from ..layers import FeaturesProjector

@dataclass
class CrossEncoderHeadOutput(BaseModelOutput):
    batch_mapping: torch.Tensor = None
    cross_input_ids: torch.Tensor = None
    cross_embeddings: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    hidden_states: Tuple[torch.Tensor] = None
    attentions: Tuple[torch.Tensor] = None
    logits: torch.Tensor = None
    cls_mask: torch.Tensor = None


class BaseCrossEncoderHead(PreTrainedModel):
    config_class = GLiClassModelConfig
    """
    Base class for cross-encoding heads.
    """
    def __init__(self, config):
        if isinstance(config.cross_encoder_config, dict):
            cross_cfg = CrossEncoderHeadConfig(**config.cross_encoder_config)
        else:
            cross_cfg = config.cross_encoder_config
        config.cross_encoder_config = cross_cfg
        super().__init__(config)
        self.config = config
        self.active_layers = config.cross_encoder_config.active_layers
        self.z_steps = config.cross_encoder_config.z_steps
        self.inner_batch_size = config.cross_encoder_config.inner_batch_size
        self.scorer = ScorerDot(hidden_size=self.config.hidden_size)

        self.text_projector = FeaturesProjector(config)
        self.classes_projector = FeaturesProjector(config)
        self.dropaut = torch.nn.Dropout(0.2)

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
            "batch_mapping": pairs,
            "cross_embeddings": cross_embeds,
            "cross_input_ids": cross_ids,
            "cross_attention_mask": cross_attention_mask,
        }

    def _extract_class_features(self, token_embeds: torch.Tensor, batch_mapping: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embeds: [num_pairs, seq_len, hidden_dim]
            batch_mapping: [num_pairs, 5] — (batch_idx, cls_start, cls_end, text_start, text_end)
        Returns:
            cls_features:  [batch_size, max_cls_len, hidden_dim]
            cls_mask:      [batch_size, max_cls_len]
            text_features: [batch_size, max_cls_len, hidden_dim]
            text_mask:     [batch_size, max_cls_len]
        """
        device = token_embeds.device
        hidden_dim = token_embeds.size(-1)

        batch2cls = {}
        batch2txt = {}
        for i in range(token_embeds.size(0)):
            b = int(batch_mapping[i, 0])
            cls_len = int(batch_mapping[i, 2] - batch_mapping[i, 1])
            if token_embeds.size(1) > 1:
                cls_repr = token_embeds[i, 1]
            else:
                cls_repr = token_embeds[i, 0]
            txt_idx = 1 + cls_len
            if token_embeds.size(1) > txt_idx:
                txt_repr = token_embeds[i, txt_idx]
            else:
                txt_repr = torch.zeros(hidden_dim, dtype=token_embeds.dtype, device=device)
            batch2cls.setdefault(b, []).append(cls_repr)
            batch2txt.setdefault(b, []).append(txt_repr)

        batch_size = max(batch2cls.keys()) + 1
        max_cls_len = max(len(v) for v in batch2cls.values())

        cls_features = torch.zeros(batch_size, max_cls_len, hidden_dim, dtype=token_embeds.dtype, device=device)
        text_features = torch.zeros(batch_size, max_cls_len, hidden_dim, dtype=token_embeds.dtype, device=device)
        cls_mask = torch.zeros(batch_size, max_cls_len, dtype=torch.long, device=device)
        text_mask = torch.zeros(batch_size, max_cls_len, dtype=torch.long, device=device)

        for b, cls_tokens in batch2cls.items():
            n = len(cls_tokens)
            cls_features[b, :n] = torch.stack(cls_tokens)
            cls_mask[b, :n] = 1
            text_features[b, :n] = torch.stack(batch2txt[b])
            text_mask[b, :n] = 1

        return cls_features, cls_mask, text_features, text_mask

    @abstractmethod
    def _forward(
        self,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inner_batch_size: int = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> CrossEncoderHeadOutput:
        """
        Forward pass for the cross-encoder head.

        Args:
            token_embeds: [batch_size, seq_len, hidden_dim]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            inner_batch_size: Optional inner batch size for processing.
            output_hidden_states: Whether to return hidden states.
            output_attentions: Whether to return attentions.
            return_dict: Whether to return a dictionary output.

        Returns:
            CrossEncoderHeadOutput containing the results of the forward pass.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def forward(
        self,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inner_batch_size: int = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> CrossEncoderHeadOutput:
        """
        Main forward method that calls the _forward implementation.
        """
        outputs =  self._forward(
            token_embeds=token_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            inner_batch_size=inner_batch_size,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        cls_features, cls_mask, text_features, text_mask = self._extract_class_features(
            token_embeds=outputs.cross_embeddings,
            batch_mapping=outputs.batch_mapping,
        )
        text_features = self.dropaut(self.text_projector(text_features))

        scores = self.scorer(cls_features, text_features)

        return CrossEncoderHeadOutput(
            last_hidden_state=outputs.last_hidden_state,
            batch_mapping=outputs.batch_mapping,
            cls_mask=cls_mask,
            cross_input_ids=outputs.cross_input_ids,
            logits=scores,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )