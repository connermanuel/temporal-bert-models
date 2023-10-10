"""
Implements a modification of the self-attention layer that interweaves temporal context using orthogonal layers.
Uses a custom parallel batch linear layer to speed up computation.
Based on the Huggingface transformers library, using BertSelfAttention as a superclass.

TODO: Reconstruct using base models instead of reusing BertSelfAttention as superclass. See https://huggingface.co/docs/transformers/v4.26.1/en/philosophy
HF modeling utils: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
HF modeling bert: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
"""

from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertSelfAttention,
    BertLayer,
    BertEncoder,
    BertModel,
    BertForMaskedLM,
    BertPreTrainedModel,
)
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers import BertConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal

from tempo_models.models.multi_head_linear import MultiHeadLinear

import math
import warnings
from typing import Optional, Tuple, Union
from copy import deepcopy
import os

TIMESTAMP_PAD = 2


class OrthogonalConfig(BertConfig):
    def __init__(self, n_contexts=2, alpha=0, init_temporal_weights=True, **kwargs):
        super().__init__(**kwargs)
        self.n_contexts = n_contexts
        self.alpha = alpha
        self.init_temporal_weights = init_temporal_weights

class BertOrthogonalSelfAttention(BertSelfAttention):
    def __init__(self, config):
        self.n_contexts = config.n_contexts
        super().__init__(config)
        self.query_time_layers = nn.ModuleList(
            [
                orthogonal(
                    MultiHeadLinear(
                        self.num_attention_heads,
                        self.attention_head_size,
                        self.attention_head_size,
                        bias=False,
                    )
                )
                for _ in range(self.n_contexts + 2)
            ]
        )
        self.key_time_layers = nn.ModuleList(
            [
                orthogonal(
                    MultiHeadLinear(
                        self.num_attention_heads,
                        self.attention_head_size,
                        self.attention_head_size,
                        bias=False,
                    )
                )
                for _ in range(self.n_contexts + 2)
            ]
        )

    def init_temporal_weights(self):
        """Initializes all of the orthogonal layers to have the same weights, since O @ O^T = I."""
        state_dict = self.query_time_layers[0].state_dict()
        for query_layer, key_layer in zip(self.query_time_layers, self.key_time_layers):
            query_layer.load_state_dict(state_dict)
            key_layer.load_state_dict(state_dict)

    def construct_time_matrix(
        self,
        original_layer: torch.Tensor,
        time_layers: nn.ModuleList,
        timestamps: torch.tensor,
    ):
        """
        Multiply rows of the input matrix with the corresponding time layers.

        Input:
            original_layer: tensor of shape (batch_size, seq_len, attention_full_size)
            time_layers: Nested module list of shape (num_timestamps, num_attention_heads), each module is a linear layer
            timestamps: tensor of shape (batch_size, seq_len)
        Output:
            temporal_conditioned_layer: tensor of shape (batch_size, num_attention_heads, seq_len, attention_head_size)
        """

        # TODO: Optimize
        original_layer = self.transpose_for_scores(original_layer)
        timestamp_vals = torch.unique(timestamps)
        masks = [
            torch.unsqueeze(timestamps == val, 1)[..., None] for val in timestamp_vals
        ]
        temporal_conditioned_layers = [
            time_layers[val + TIMESTAMP_PAD](original_layer) * mask
            for val, mask in zip(timestamp_vals, masks)
        ]
        temporal_conditioned_layer = torch.stack(
            temporal_conditioned_layers, dim=0
        ).sum(dim=0)

        return temporal_conditioned_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # TODO: change device to be passed in as an arg
        # TODO: implement compatibility for temporal seq2seq decoding

        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        timed_query_layer = self.construct_time_matrix(
            mixed_query_layer, self.query_time_layers, timestamps
        )
        timed_key_layer = self.construct_time_matrix(
            mixed_key_layer, self.key_time_layers, timestamps
        )
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            timed_query_layer, timed_key_layer.transpose(-1, -2)
        )
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class BertOrthogonalAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = BertOrthogonalSelfAttention(config)

    def forward(
        self,
        hidden_states,
        timestamps,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            timestamps,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class BertOrthogonalLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertOrthogonalAttention(config)

    def forward(
        self,
        hidden_states,
        timestamps,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            timestamps,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs


class BertOrthogonalEncoder(BertEncoder):
    def __init__(self, config, init_temporal_weights=True):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [BertOrthogonalLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # if init_temporal_weights:
        #     self.init_temporal_weights()

    def forward(
        self,
        hidden_states,
        timestamps,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    timestamps,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    timestamps,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def init_temporal_weights(self):
        [layer.attention.self.init_temporal_weights() for layer in self.layer]


class BertOrthogonalModel(BertModel):
    def __init__(self, config, add_pooling_layer=True, init_temporal_weights=True):
        super().__init__(
            config, add_pooling_layer
        )  # initializes embeddings and creates init_weights
        self.encoder = BertOrthogonalEncoder(
            config, init_temporal_weights=init_temporal_weights
        )
        self.n_contexts = config.n_contexts
        self.post_init()
        self.init_temporal_weights()

    def forward(
        self,
        input_ids=None,
        timestamps=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:


            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if timestamps is None:
            raise ValueError(
                f"You need to pass in a list of timestamps, ranging from 0 to {self.n_contexts - 1}"
            )
        elif timestamps.shape != input_ids.shape:
            raise ValueError(
                f"Timestamps not properly generated: must be same shape as input ids"
            )

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            timestamps,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def init_temporal_weights(self):
        self.encoder.init_temporal_weights()


class BertForOrthogonalMaskedLM(BertForMaskedLM):
    def __init__(self, config, init_temporal_weights=True):
        super().__init__(config)
        self.bert = BertOrthogonalModel(
            config, add_pooling_layer=False, init_temporal_weights=init_temporal_weights
        )
        self.post_init()
        self.init_temporal_weights()

        self.alpha = config.alpha

    def forward(
        self,
        input_ids=None,
        timestamps=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert (
            "lm_labels" not in kwargs
        ), "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            timestamps=timestamps,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def init_temporal_weights(self):
        self.bert.init_temporal_weights()


class BertForOrthogonalSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, init_temporal_weights=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = config.alpha

        self.bert = BertOrthogonalModel(
            config, init_temporal_weights=init_temporal_weights
        )
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()
        self.init_temporal_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            timestamps=timestamps,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def init_temporal_weights(self):
        self.bert.init_temporal_weights()
