import os
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import (logging)
from transformers.models.auto import AutoModel
from .config import GLiClassModelConfig
from .layers import FeaturesProjector, LstmSeq2SeqEncoder, BiEncoderProjector, LayerwiseAttention
from .poolings import POOLING2OBJECT
from .scorers import SCORER2OBJECT
from .loss_functions import focal_loss_with_logits, sequence_contrastive_loss
from .utils import is_module_available, MissedPackageException

IS_LLM2VEC = is_module_available('llm2vec')
IS_PEFT = is_module_available('peft')
IS_TURBOT5 = is_module_available('turbot5')
IS_FLASHDEBERTA = is_module_available('flashdeberta')

logger = logging.get_logger(__name__)

if IS_LLM2VEC:
    from llm2vec.models import MistralBiModel, LlamaBiModel, GemmaBiModel, Qwen2BiModel
    DECODER_MODEL_MAPPING = {
        "MistralConfig": MistralBiModel,
        "LlamaConfig": LlamaBiModel,
        "GemmaConfig": GemmaBiModel,
        "Qwen2Config": Qwen2BiModel
    }
else:
    DECODER_MODEL_MAPPING = {}

if IS_TURBOT5:
    from turbot5.model.modeling import T5EncoderModel
else:
    from transformers import T5EncoderModel

if IS_FLASHDEBERTA:
    from flashdeberta import FlashDebertaV2Model as DebertaV2Model
else:
    from transformers import DebertaV2Model

if IS_PEFT:
    from peft import LoraConfig, get_peft_model

@dataclass
class GLiClassOutput(SequenceClassifierOutput):
    text_embeddings: Optional[torch.Tensor] = None
    class_embeddings: Optional[torch.Tensor] = None

class GLiClassPreTrainedModel(PreTrainedModel):
    config_class = GLiClassModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]
    
    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.encoder_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    nn.init.normal_(param.data, mean=0.0, std=std)
                elif 'bias' in name:
                    param.data.zero_()
    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


class GLiClassBaseModel(nn.Module):#):
    def __init__(self, config: GLiClassModelConfig, device='cpu', **kwargs):
        super().__init__()
        self.config = config
        self.text_projector = FeaturesProjector(config)
        self.classes_projector = FeaturesProjector(config)
        
        if config.pooling_strategy not in POOLING2OBJECT:
            raise NotImplementedError(f"{config.pooling_strategy} is not implemented pooling type.")
        else:
            self.pooler = POOLING2OBJECT[config.pooling_strategy]()

        if config.pooling_strategy not in POOLING2OBJECT:
            raise NotImplementedError(f"{config.scorer_type} is not implemented. Choose one of this: 'dot', 'weighted-dot'")
        else:
            self.scorer = SCORER2OBJECT[config.scorer_type](config.hidden_size)

        if config.use_lstm:
            self.lstm = LstmSeq2SeqEncoder(config.hidden_size, config.hidden_size//2, bidirectional=True)
        
        if config.squeeze_layers:
            self.layer_wise_attention = LayerwiseAttention(config.encoder_config.num_hidden_layers,
                                                           config.encoder_config.hidden_size)
            
        drop_out = getattr(config.encoder_config, "cls_dropout", None)
        if drop_out is None:
            if hasattr(self.config.encoder_config, 'hidden_dropout_prob'):
                drop_out = self.config.encoder_config.hidden_dropout_prob 
            elif hasattr(self.config.encoder_config, 'dropout_rate'):
                drop_out = self.config.encoder_config.dropout_rate
            else:
                drop_out = 0.15
        # self.dropout = StableDropout(drop_out)
        self.dropout = nn.Dropout(drop_out)

        
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.epsilon = 1e-8
        self.vocab_size = config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.num_labels = -1
        
        self.device = torch.device(device)

    def _extract_class_features(self, token_embeds, input_ids, attention_mask):
        batch_size, sequence_length, embed_dim = token_embeds.shape

        class_token_mask = input_ids == self.config.class_token_index
        num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)

        max_embed_dim = num_class_tokens.max()
                
        aranged_class_idx = torch.arange(max_embed_dim, 
                                            dtype=attention_mask.dtype, 
                                            device=token_embeds.device).expand(batch_size, -1)
        
        batch_indices, target_class_idx = torch.where(aranged_class_idx<num_class_tokens)
        _, class_indices = torch.where(class_token_mask)
        if not self.config.embed_class_token:
            class_indices+=1

        classes_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
        )
        classes_embedding_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=token_embeds.device
        )

        classes_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]
        classes_embedding_mask[batch_indices, target_class_idx] = 1
        
        #tokens features extraction
        if self.config.extract_text_features:
            text_token_mask = input_ids == self.config.text_token_index

            _, text_token_indices = torch.where(text_token_mask)

            text_lengths = input_ids.shape[-1]-text_token_indices

            max_text_length = text_lengths.max()

            text_tokens_embeddings = torch.zeros(
                batch_size, max_text_length, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
            )
            text_tokens_mask = torch.zeros(
                batch_size, max_text_length, dtype=attention_mask.dtype, device=token_embeds.device
                    )

            aranged_token_idx = torch.arange(input_ids.shape[-1], device=token_embeds.device).unsqueeze(0)

            batch_idx, text_token_idx = torch.where(aranged_token_idx>=text_token_indices.unsqueeze(1))
            _, target_text_idx = torch.where(aranged_token_idx<text_lengths.unsqueeze(1))

            text_tokens_embeddings[batch_idx, target_text_idx] = token_embeds[batch_idx, text_token_idx]
            text_tokens_mask[batch_idx, target_text_idx] = attention_mask[batch_idx, text_token_idx]

        else:
            text_tokens_embeddings = token_embeds
            text_tokens_mask = attention_mask
        return classes_embedding, classes_embedding_mask, text_tokens_embeddings, text_tokens_mask

    def get_loss(self, logits, labels, classes_embedding=None, classes_embedding_mask=None):
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                all_losses = focal_loss_with_logits(logits, labels, 
                                    self.config.focal_loss_alpha, self.config.focal_loss_gamma, self.config.focal_loss_reduction)
                if classes_embedding_mask is not None:
                    all_losses = all_losses * classes_embedding_mask.float()
                loss = all_losses.mean()

            if self.config.contrastive_loss_coef>0 and classes_embedding is not None:
                contrastive_loss = sequence_contrastive_loss(classes_embedding, classes_embedding_mask)
                loss = loss+contrastive_loss*self.config.contrastive_loss_coef
        return loss
    
class GLiClassUniEncoder(GLiClassBaseModel):
    def __init__(self, config: GLiClassModelConfig, from_pretrained = False):
        super().__init__(config)
        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder model name to use it as a backbone.")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)

        config_name = config.encoder_config.__class__.__name__

        if config_name in DECODER_MODEL_MAPPING:
            if not IS_LLM2VEC:
                raise MissedPackageException(f"The llm2vec package must be installed to use this decoder model: {config_name}")
            else:
                print('Loading decoder model using LLM2Vec...')
                ModelClass = DECODER_MODEL_MAPPING[config_name]
            decoder = True
        elif config_name in {'T5Config', 'MT5Config'}:
            decoder = False
            ModelClass = T5EncoderModel
        elif config_name in {'DebertaV2Config'}:
            decoder = False
            ModelClass = DebertaV2Model
        else:
            decoder = False
            ModelClass = AutoModel

        if from_pretrained:
            self.encoder_model = ModelClass.from_pretrained(
                config.encoder_model_name
            )
        else:
            if decoder:
                self.encoder_model = ModelClass(config.encoder_config)
            else:
                if config_name in {'T5Config', 'MT5Config', 'DebertaV2Config'}:
                    self.encoder_model = ModelClass._from_config(
                        config.encoder_config
                    )
                else:
                    self.encoder_model = ModelClass.from_config(
                        config.encoder_config
                    )

        adapter_config_file = Path(config.encoder_model_name) / "adapter_config.json"

        if adapter_config_file.exists():
            if not IS_PEFT:
                warnings.warn(f"Adapter configs were detected, if you want to apply them you need to install peft package.")
            else:
                adapter_config = LoraConfig.from_pretrained(config.encoder_model_name)
                self.encoder_model = get_peft_model(self.encoder_model, adapter_config)

    def _extract_class_features(self, token_embeds, input_ids, attention_mask):
        batch_size, seq_len, hidden_dim = token_embeds.shape
        device = token_embeds.device

        # ------------------------------------------------------------------ class tokens
        class_token_mask = input_ids == self.config.class_token_index
        num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)

        max_embed_dim = num_class_tokens.max()
                
        aranged_class_idx = torch.arange(max_embed_dim, 
                                            dtype=attention_mask.dtype, 
                                            device=device).expand(batch_size, -1)
        batch_indices, target_class_idx = torch.where(aranged_class_idx<num_class_tokens)
        # ------------------------------------------------------------------------> new logic
        # class_token_mask = input_ids == self.config.class_token_index
        # class_batch_indices, class_indices = torch.where(class_token_mask)
        # text_token_mask = input_ids == self.config.text_token_index
        # print("num_class_tokens ", num_class_tokens)

        # text_batch_indices, text_token_indices = torch.where(text_token_mask)
        # print(f"Class indices: {class_indices}")
        # print(f"Batch indices: {class_batch_indices}")
        # print(f"Text token indices: {text_token_indices}")
        # print(f"Batch indices: {text_batch_indices}")
        # class_indices = class_indices
        # class_batch_indices = class_batch_indices
        # print(f"Class indices: {class_indices}")
        # insert_pos = []

        # print("input_ids ", input_ids)
        # print("attention_mask ", attention_mask)

        # ######
        # text_token_mask = input_ids == self.config.text_token_index

        # bos_text_batch_indices, bos_text_indices = torch.where(text_token_mask)
        # bos_text_indices = bos_text_indices
        # reversed_attention = attention_mask.flip(dims=[1])
        # last_nonzero_idx = seq_len - reversed_attention.float().argmax(dim=1) - 1
        # eos_mask = torch.zeros_like(attention_mask, dtype=torch.bool).scatter(1, last_nonzero_idx.unsqueeze(1), True)
        # eos_text_batch_indices, eos_text_indices = torch.where(eos_mask)
        # print(f"eos_text_indices: {eos_text_indices}")
        # ######
        # batch2classes = {}
        # for b in range(batch_size):
        #     if b not in batch2classes:
        #         batch2classes[b] = torch.tensor([], dtype=torch.int64, device=device)
        #     for class_idx in class_indices[class_batch_indices==b]:
        #         batch2classes[b] = torch.cat((batch2classes[b], class_idx.unsqueeze(0)), dim=0)
        #     batch2classes[b] = torch.cat((batch2classes[b], bos_text_indices[b].unsqueeze(0)), dim=0)
        # print(f"Batch to classes mapping: {batch2classes}")
        # self.inner_batch_size = 4

        # pairs = None
        # for b in range(batch_size):
        #     class_start = batch2classes[b][:-1]
        #     class_end = batch2classes[b][1:]
        #     text_start = torch.full_like(class_start, bos_text_indices[b])
        #     text_end = torch.full_like(class_start, eos_text_indices[b])
        #     batch_ids = torch.full_like(class_start, b)
        #     pair = torch.stack((batch_ids, class_start, class_end, text_start, text_end), dim=1)
        #     if pairs is None:
        #         pairs = pair
        #     else:
        #         pairs = torch.cat((pairs, pair), dim=0)

        # print(f"Pairs: {pairs}")

        # max_seq_len = pairs[:, 2] - pairs[:, 1] + pairs[:, 4] - pairs[:, 3] + 2
        # n_pairs = pairs.shape[0]
        # max_seq_len = max_seq_len.max().item()
        # zero_embeddings = torch.zeros(
        #     n_pairs, max_seq_len, hidden_dim, dtype=token_embeds.dtype, device=device
        # )
        # zero_input_ids = torch.zeros(
        #     n_pairs, max_seq_len, dtype=input_ids.dtype, device=device
        # )
        # zero_attention_mask = torch.zeros(
        #     n_pairs, max_seq_len, dtype=attention_mask.dtype, device=device
        # )

        # for i in range(n_pairs):
        #     b, cls_start, cls_end, txt_start, txt_end = pairs[i]

        #     cls_len = int(cls_end - cls_start)
        #     txt_len = int(txt_end - txt_start)

        #     bos_pos = 0
        #     cls_pos = bos_pos + 1
        #     txt_pos = cls_pos + cls_len
        #     eos_pos = txt_pos + txt_len

        #     zero_embeddings[i, bos_pos, :] = token_embeds[b, 0, :]  # BOS
        #     zero_embeddings[i, cls_pos:cls_pos + cls_len, :] = token_embeds[b, cls_start:cls_end, :]
        #     zero_embeddings[i, txt_pos:txt_pos + txt_len, :] = token_embeds[b, txt_start:txt_end, :]
        #     zero_embeddings[i, eos_pos, :] = token_embeds[b, txt_end, :]  # EOS

        #     zero_input_ids[i, bos_pos] = input_ids[b, 0]  # BOS
        #     zero_input_ids[i, cls_pos:cls_pos + cls_len] = input_ids[b, cls_start:cls_end]
        #     zero_input_ids[i, txt_pos:txt_pos + txt_len] = input_ids[b, txt_start:txt_end]
        #     zero_input_ids[i, eos_pos] = input_ids[b, txt_end] # EOSs
            
        # zero_attention_mask = (zero_input_ids > 0).long()
        # print(f"Zero input ids shape: {zero_input_ids}")
        # print(f"Zero embeddings shape: {zero_embeddings.shape}")
        # print(f"Zero attention mask shape: {zero_attention_mask}")
        # ------------------------------------------------------------------------> new logic
        if not self.config.embed_class_token:
            class_indices+=1

        classes_embedding = torch.zeros(
            batch_size, max_embed_dim, hidden_dim, dtype=token_embeds.dtype, device=device
        )
        classes_embedding_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=device
        )
        classes_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]
        classes_embedding_mask[batch_indices, target_class_idx] = 1
        
        # ------------------------------------------------------------------ text tokens
        if self.config.extract_text_features:
            text_token_mask = input_ids == self.config.text_token_index

            _, text_token_indices = torch.where(text_token_mask)

            reversed_attention = attention_mask.flip(dims=[1])
            last_nonzero_idx = seq_len - reversed_attention.float().argmax(dim=1) - 1

            eos_mask = torch.zeros_like(attention_mask, dtype=torch.bool).scatter(1, last_nonzero_idx.unsqueeze(1), True)
            _, eos_token_indices = torch.where(eos_mask)

            text_lengths = eos_token_indices - text_token_indices
            max_text_seq_len = text_lengths.max()

            text_tokens_embeddings = torch.zeros(
                batch_size, max_text_seq_len, hidden_dim, dtype=token_embeds.dtype, device=device
            )
            text_tokens_mask = torch.zeros(
                batch_size, max_text_seq_len, dtype=attention_mask.dtype, device=device
            )

            token_index_range = torch.arange(seq_len, device=device).unsqueeze(0)
            text_span_mask = (token_index_range >= text_token_indices.unsqueeze(1)) & (token_index_range < eos_token_indices.unsqueeze(1))

            batch_idx, src_token_idx = torch.where(text_span_mask)
            target_text_idx = src_token_idx - text_token_indices[batch_idx]

            text_tokens_embeddings[batch_idx, target_text_idx, :] = token_embeds[batch_idx, src_token_idx, :]
            text_tokens_mask[batch_idx, target_text_idx] = attention_mask[batch_idx, src_token_idx]

        else:
            text_tokens_embeddings = token_embeds
            text_tokens_mask = attention_mask
        return classes_embedding, classes_embedding_mask, text_tokens_embeddings, text_tokens_mask
    
    def process_encoder_output(self, input_ids, attention_mask, encoder_layer, labels = None):
        classes_embedding, classes_embedding_mask, text_token_embeddings, text_mask = self._extract_class_features(encoder_layer, 
                                                                                                            input_ids, attention_mask)
        if self.config.use_lstm:
            text_token_embeddings = self.lstm(text_token_embeddings, text_mask)
        
        pooled_output = self.pooler(text_token_embeddings)
        pooled_output = self.text_projector(pooled_output)
        pooled_output = self.dropout(pooled_output)
        if self.config.normalize_features:
            pooled_output = pooled_output / (pooled_output.norm(p=2, dim=-1, keepdim=True)+self.epsilon)

        classes_embedding = self.classes_projector(classes_embedding)
        if self.config.normalize_features:
            classes_embedding = classes_embedding / (classes_embedding.norm(p=2, dim=-1, keepdim=True)+self.epsilon)

        logits = self.scorer(pooled_output, classes_embedding)

        if self.config.normalize_features:
            logits = logits*self.logit_scale.to(classes_embedding.device)
        
        loss = self.get_loss(logits, labels, classes_embedding, classes_embedding_mask)
        return (logits, loss, pooled_output, classes_embedding)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_text_embeddings: Optional[bool] = None,
        output_class_embeddings:  Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, GLiClassOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.squeeze_layers or self.config.layer_wise:
            output_hidden_states = True
            return_dict = True

        outputs = self.encoder_model(
            input_ids,
            attention_mask=attention_mask,
            # inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        # hidden_states = outputs.last_hidden_state
        # rel_embeddings = self.encoder_model.encoder.get_rel_embedding()
        # attention_mask = self.encoder_model.encoder.get_attention_mask(attention_mask)
        # relative_pos = self.encoder_model.encoder.get_rel_pos(hidden_states)

        # for _ in range(0):
        #     for i in reversed(range(1, 2)):
        #         print(i)
        #         hidden_states, _ = self.encoder_model.encoder.layer[-i](
        #             hidden_states,
        #             attention_mask,
        #             relative_pos=relative_pos,
        #             rel_embeddings=rel_embeddings,
        #         )
        # from transformers.modeling_outputs import BaseModelOutput
        # outputs = BaseModelOutput(
        #     last_hidden_state=hidden_states,
        #     hidden_states=None,
        #     attentions=None
        # )
        if self.config.layer_wise and labels is not None:
            hidden_states = outputs.hidden_states
            loss = 0
            for encoder_layer in hidden_states:
                logits, layer_loss, pooled_output, classes_embedding = self.process_encoder_output(input_ids, attention_mask, encoder_layer, labels)
                loss+=layer_loss
        else:
            if self.config.encoder_layer_id==-1:
                if self.config.squeeze_layers:
                    encoder_layer = self.layer_wise_attention(outputs.hidden_states)
                else:
                    encoder_layer = outputs[0]
            else:
                encoder_layer = outputs.hidden_states[self.config.encoder_layer_id]
            logits, loss, pooled_output, classes_embedding = self.process_encoder_output(input_ids, attention_mask, encoder_layer, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return GLiClassOutput(
            loss=loss, logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
            text_embeddings= pooled_output if output_text_embeddings else None,
            class_embeddings= classes_embedding if output_class_embeddings else None,
        )

from .cross_encoder_heads.models.deberta_v2 import DebertaV2CrossEncoderHead
from .cross_encoder_heads.config import CrossEncoderHeadConfig

class GLiClassUniCrossEncoder(GLiClassBaseModel):
    def __init__(self, config: GLiClassModelConfig, from_pretrained = False):
        super().__init__(config)
        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder model name to use it as a backbone.")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)

        config_name = config.encoder_config.__class__.__name__

        if config_name in DECODER_MODEL_MAPPING:
            if not IS_LLM2VEC:
                raise MissedPackageException(f"The llm2vec package must be installed to use this decoder model: {config_name}")
            else:
                print('Loading decoder model using LLM2Vec...')
                ModelClass = DECODER_MODEL_MAPPING[config_name]
            decoder = True
        elif config_name in {'T5Config', 'MT5Config'}:
            decoder = False
            ModelClass = T5EncoderModel
        elif config_name in {'DebertaV2Config'}:
            decoder = False
            ModelClass = DebertaV2Model
        else:
            decoder = False
            ModelClass = AutoModel

        if from_pretrained:
            self.encoder_model = ModelClass.from_pretrained(
                config.encoder_model_name
            )
        else:
            if decoder:
                self.encoder_model = ModelClass(config.encoder_config)
            else:
                if config_name in {'T5Config', 'MT5Config', 'DebertaV2Config'}:
                    self.encoder_model = ModelClass._from_config(
                        config.encoder_config
                    )
                else:
                    self.encoder_model = ModelClass.from_config(
                        config.encoder_config
                    )

        adapter_config_file = Path(config.encoder_model_name) / "adapter_config.json"

        if adapter_config_file.exists():
            if not IS_PEFT:
                warnings.warn(f"Adapter configs were detected, if you want to apply them you need to install peft package.")
            else:
                adapter_config = LoraConfig.from_pretrained(config.encoder_model_name)
                self.encoder_model = get_peft_model(self.encoder_model, adapter_config)
        self.scorer = None
        self.cross_encoder_head = DebertaV2CrossEncoderHead(
            self.config,
            self.encoder_model,
        )
                
    
    def process_encoder_output(self, input_ids, attention_mask, encoder_layer, labels = None):
        
        outputs = self.cross_encoder_head(
            token_embeds=encoder_layer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            )
        logits = outputs.logits
        classes_embedding_mask = outputs.cls_mask
        classes_embedding = None
        
        loss = self.get_loss(logits, labels, classes_embedding, classes_embedding_mask)
        return (logits, loss)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_text_embeddings: Optional[bool] = None,
        output_class_embeddings:  Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, GLiClassOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.squeeze_layers or self.config.layer_wise:
            output_hidden_states = True
            return_dict = True

        outputs = self.encoder_model(
            input_ids,
            attention_mask=attention_mask,
            # inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        if self.config.layer_wise and labels is not None:
            hidden_states = outputs.hidden_states
            loss = 0
            for encoder_layer in hidden_states:
                logits, layer_loss, pooled_output, classes_embedding = self.process_encoder_output(input_ids, attention_mask, encoder_layer, labels)
                loss+=layer_loss
        else:
            if self.config.encoder_layer_id==-1:
                if self.config.squeeze_layers:
                    encoder_layer = self.layer_wise_attention(outputs.hidden_states)
                else:
                    encoder_layer = outputs[0]
            else:
                encoder_layer = outputs.hidden_states[self.config.encoder_layer_id]
            logits, loss = self.process_encoder_output(input_ids, attention_mask, encoder_layer, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return GLiClassOutput(
            loss=loss, logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
            text_embeddings= pooled_output if output_text_embeddings else None,
            class_embeddings= classes_embedding if output_class_embeddings else None,
        )

class GLiClassEncoderDecoder(GLiClassBaseModel):
    def __init__(self, config: GLiClassModelConfig, from_pretrained = False):
        super().__init__(config)
        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder model name to use it as a backbone.")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)

        if not config.encoder_config.is_encoder_decoder:
            raise ValueError("You need to choose encoder-decoder model as a backbone.")
        
        if from_pretrained:
            self.encoder_decoder_model = AutoModel.from_pretrained(
                config.encoder_model_name
            )
        else:
            self.encoder_decoder_model = AutoModel.from_config(
                config.encoder_config
            )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        class_input_ids: Optional[torch.Tensor] = None,
        class_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_text_embeddings: Optional[bool] = None,
        output_class_embeddings:  Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder_decoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=class_input_ids,
            decoder_attention_mask=class_attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        text_token_embeddings = outputs.encoder_last_hidden_state
        decoder_token_embeddings = outputs.last_hidden_state
        classes_embedding, classes_embedding_mask, _, _ = self._extract_class_features(decoder_token_embeddings, 
                                                                                class_input_ids, class_attention_mask)
        
        if self.config.use_lstm:
            text_token_embeddings = self.lstm(text_token_embeddings, attention_mask)
        
        pooled_output = self.pooler(text_token_embeddings)
        pooled_output = self.text_projector(pooled_output)
        pooled_output = self.dropout(pooled_output)
        if self.config.normalize_features:
            pooled_output = nn.functional.normalize(pooled_output, p=2, dim=-1, eps=self.epsilon)

        classes_embedding = self.classes_projector(classes_embedding)
        if self.config.normalize_features:
            classes_embedding = nn.functional.normalize(classes_embedding, p=2, dim=-1, eps=self.epsilon)

        logits = self.scorer(pooled_output, classes_embedding)

        if self.config.normalize_features:
            logits = logits*self.logit_scale.to(classes_embedding.device)
        
        loss = self.get_loss(logits, labels, classes_embedding, classes_embedding_mask)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return GLiClassOutput(
            loss=loss, logits=logits, 
            hidden_states = outputs.hidden_states, 
            attentions = outputs.attentions,
            text_embeddings = pooled_output if output_text_embeddings else None,
            class_embeddings = classes_embedding if output_class_embeddings else None,
        )
 
class GLiClassBiEncoder(GLiClassBaseModel):
    def __init__(self, config: GLiClassModelConfig, from_pretrained=False):
        super().__init__(config)
        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder model name to use it as a backbone.")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)

        if config.label_model_config is None:
            if config.label_model_name is None:
                raise ValueError("You need to specify label model name to use it as a backbone.")
            config.label_model_config = AutoConfig.from_pretrained(config.label_model_name)

        def initialize_encoder(configs, model_name, from_pretrained):
            if from_pretrained:
                return AutoModel.from_pretrained(model_name)
            else:
                return AutoModel.from_config(configs)
        self.encoder_model = initialize_encoder(config.encoder_config, config.encoder_model_name, from_pretrained)
        self.label_encoder = initialize_encoder(config.label_model_config, config.label_model_name, from_pretrained)
        self.biencoder_projector = BiEncoderProjector(config)

    def pool_outputs(self, encoder_outputs):
        text_embeddings = self.pooler(encoder_outputs[0])
        text_embeddings = self.text_projector(text_embeddings)
        text_embeddings = self.dropout(text_embeddings)
        if self.config.normalize_features:
            text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=-1, eps=self.epsilon)
        return text_embeddings

    def encode_text(self, input_ids, attention_mask):
        outputs = self.encoder_model(input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1))
        text_embeddings = self.pool_outputs(outputs)
        return text_embeddings

    def encode_classes(self, class_input_ids, class_attention_mask, labels_mask=None):
        batch_size = class_input_ids.shape[0]
        num_classes = class_input_ids.shape[1]
        if labels_mask is not None:
            batch_indices, indices = torch.where(labels_mask==1)
            selected_input_ids = class_input_ids[batch_indices, indices]
            selected_attention_mask = class_attention_mask[batch_indices, indices]

            outputs = self.label_encoder(selected_input_ids, attention_mask=selected_attention_mask)
            class_embeddings_filtered = self.pooler(outputs[0])

            class_embeddings = torch.zeros(batch_size, num_classes, class_embeddings_filtered.shape[-1], 
                                                                    dtype=class_embeddings_filtered.dtype, 
                                                                    device=class_embeddings_filtered.device)

            class_embeddings[batch_indices, indices] = class_embeddings_filtered
        else:  
            class_input_ids = class_input_ids.view(-1, class_input_ids.shape[-1])
            class_attention_mask = class_attention_mask.view(-1, class_input_ids.shape[-1])
            outputs = self.label_encoder(class_input_ids, attention_mask=class_attention_mask)
            class_embeddings = self.pooler(outputs[0])
            class_embeddings = class_embeddings.reshape(batch_size, num_classes, -1)
        class_embeddings = self.biencoder_projector(class_embeddings)
        class_embeddings = self.classes_projector(class_embeddings)
        if self.config.normalize_features:
            class_embeddings = nn.functional.normalize(class_embeddings, p=2, dim=-1, eps=self.epsilon)
        return class_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        class_input_ids: Optional[torch.Tensor] = None,
        class_attention_mask: Optional[torch.Tensor] = None,
        labels_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_text_embeddings: Optional[bool] = None,
        output_class_embeddings:  Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_embeddings = self.encode_text(input_ids, attention_mask)
        class_embeddings = self.encode_classes(class_input_ids, class_attention_mask, labels_mask)
        logits = self.scorer(text_embeddings, class_embeddings) * self.logit_scale.to(class_embeddings.device)

        if labels_mask is not None: 
            logits = torch.where(labels_mask == 0, -1e3, logits)

        loss = self.get_loss(logits, labels, classes_embedding_mask=labels_mask)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return GLiClassOutput(
            loss=loss, logits=logits, 
            text_embeddings = text_embeddings if output_text_embeddings else None,
            class_embeddings = class_embeddings if output_class_embeddings else None,
        )
 

class GLiClassBiEncoderFused(GLiClassBiEncoder):
    def __init__(self, config: GLiClassModelConfig, from_pretrained=False):
        super().__init__(config, from_pretrained)

    def encode_text(self, input_ids, attention_mask, class_embeddings, labels_mask):
        embedding_layer = self.encoder_model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        class_token_mask = input_ids==self.config.class_token_index
        batch_indices, class_token_indices = torch.where(class_token_mask)

        labels_batch_indices, labels_indices = torch.where(labels_mask==1)

        selected_class_embeddings = class_embeddings[labels_batch_indices, labels_indices]

        inputs_embeds[batch_indices, class_token_indices] = selected_class_embeddings
        encoder_outputs = self.encoder_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask.squeeze(1))
        
        post_class_embeddings = torch.zeros_like(class_embeddings)
        post_class_embeddings[labels_batch_indices, labels_indices] = encoder_outputs[0][batch_indices, class_token_indices]
        return encoder_outputs, post_class_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        class_input_ids: Optional[torch.Tensor] = None,
        class_attention_mask: Optional[torch.Tensor] = None,
        labels_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_text_embeddings: Optional[bool] = None,
        output_class_embeddings:  Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        raw_class_embeddings = self.encode_classes(class_input_ids, class_attention_mask, labels_mask)

        encoder_outputs, class_embeddings = self.encode_text(input_ids, attention_mask, raw_class_embeddings, labels_mask)
        
        text_embeddings = self.pool_outputs(encoder_outputs)

        logits = self.scorer(text_embeddings, class_embeddings) * self.logit_scale.to(class_embeddings.device)

        if labels_mask is not None: 
            logits = torch.where(labels_mask == 0, -1e3, logits)

        loss = self.get_loss(logits, labels, classes_embedding_mask=labels_mask)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return GLiClassOutput(
            loss=loss, logits=logits, 
            text_embeddings = text_embeddings if output_text_embeddings else None,
            class_embeddings = class_embeddings if output_class_embeddings else None,
        )

class GLiClassModel(GLiClassPreTrainedModel):
    def __init__(self, config, from_pretrained=False):
        super().__init__(config)
        if config.architecture_type == 'uni-encoder':
            self.model = GLiClassUniEncoder(config, from_pretrained)
        elif config.architecture_type == 'uni-cross-encoder':
            self.model = GLiClassUniCrossEncoder(config, from_pretrained)
        elif config.architecture_type == 'bi-encoder':
            self.model = GLiClassBiEncoder(config, from_pretrained)
        elif config.architecture_type == 'bi-encoder-fused':
            self.model = GLiClassBiEncoderFused(config, from_pretrained)
        elif config.architecture_type == 'encoder-decoder':
            self.model = GLiClassEncoderDecoder(config, from_pretrained)
        self.post_init()

    def get_input_embeddings(self):
        if self.config.architecture_type in {'uni-encoder', 'uni-cross-encoder'}:
            return self.model.encoder_model.get_input_embeddings()
        elif self.config.architecture_type == 'encoder-decoder':
            return self.model.encoder_decoder_model.get_input_embeddings()
        else:
            raise NotImplementedError('Getting input embeddings is not implemented for bi-encoder architecture')
        
    def set_input_embeddings(self, value):
        if self.config.architecture_type in {'uni-encoder', 'uni-cross-encoder'}:
            self.model.encoder_model.set_input_embeddings(value)
            return None
        elif self.config.architecture_type == 'encoder-decoder':
            self.model.encoder_decoder_model.set_input_embeddings(value)
        elif self.config.architecture_type in {'bi-encoder', 'bi-encoder-fused'}:
            self.model.encoder_model.set_input_embeddings(value)
        else:
            raise NotImplementedError('Setting input embeddings is not implemented for bi-encoder architecture')
        
    def tie_weights(self):
        if self.config.architecture_type in {'uni-encoder', 'uni-cross-encoder'}:
            return self.model.encoder_model.tie_weights()
        elif self.config.architecture_type == 'encoder-decoder':
            return self.model.encoder_decoder_model.tie_weights()
        elif self.config.architecture_type in {'bi-encoder', 'bi-encoder-fused'}:
            return self.model.encoder_model.tie_weights()
        else:
            raise NotImplementedError('Tie weights is not implemented for bi-encoder architecture')

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        if self.config.architecture_type in {'uni-encoder', 'uni-cross-encoder'}:
            model_embeds = self.model.encoder_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        elif self.config.architecture_type == 'encoder-decoder':
            model_embeds = self.model.encoder_decoder_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        elif self.config.architecture_type in {'bi-encoder-fused'}:
            model_embeds = self.model.encoder_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        else:
            raise NotImplementedError('Resizing is not implemented for bi-encoder architecture')
        self.config.encoder_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    
    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs