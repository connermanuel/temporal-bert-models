"""
Implements a modification of the self-attention layer that interweaves temporal context as described in Rosin & Radinsky, 2022.
Based on the Huggingface transformers library, using BertSelfAttention as a superclass.
"""

from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention, BertLayer, BertEncoder, BertEmbeddings, BertModel, BertForMaskedLM
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput
import torch.nn as nn
import torch
import math
import warnings
import joblib
import os

class BertTemporalSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.time = nn.Linear(config.hidden_size, self.all_head_size)

    def forward(
        self,
        hidden_states,
        temporal_embeddings,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        use_tempo=True,
        attention_fname=None
    ):
        mixed_query_layer = self.query(hidden_states)
        # Since the time representation augments both the key and query vectors, one time layer should
        # be constructed from original embeddings to represent the key, and one should take into account
        # the possibility of encoder padding tokens.

        # TODO: implement compatibility for temporal seq2seq decoding
        mixed_time_layer = self.time(temporal_embeddings)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        time_layer = self.transpose_for_scores(mixed_time_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" augmented by time.
        if use_tempo:
            time_layer_normalized = time_layer / time_layer.norm(2.0)
            attention_scores = torch.matmul(
                query_layer, time_layer.transpose(-1, -2)
            ).matmul(time_layer_normalized)
            try:
                attention_scores = torch.matmul(attention_scores, key_layer.transpose(-1, -2))
            except RuntimeError:
                print(attention_scores.shape)
                print(key_layer.transpose(-1, 2).shape)
                raise
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if attention_fname:
            joblib.dump(attention_scores, attention_fname)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class BertTemporalAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = BertTemporalSelfAttention(config)
    
    def forward(
        self,
        hidden_states,
        temporal_embeddings,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        use_tempo=True,
        attention_fname=None
    ):
        self_outputs = self.self(
            hidden_states,
            temporal_embeddings,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            use_tempo=use_tempo,
            attention_fname=attention_fname
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertTemporalLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertTemporalAttention(config)

    def forward(
        self,
        hidden_states,
        temporal_embeddings,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        use_tempo=True,
        attention_fname=None
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            temporal_embeddings,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            use_tempo=use_tempo,
            attention_fname=attention_fname
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

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
                use_tempo=use_tempo,
                attention_fname=attention_fname
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs


class BertTemporalEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertTemporalLayer(config) for _ in range(config.num_hidden_layers)])
    
    # This version of the BertEncoder uses the same temporal embeddings for all layers
    def forward(
        self,
        hidden_states,
        temporal_embeddings,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        use_tempo=True,
        attention_dir=None
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        layer_attention_fname = None
        if attention_dir and not os.path.exists(attention_dir):
            os.makedirs(attention_dir)
        for i, layer_module in enumerate(self.layer):
            if attention_dir:
                layer_attention_fname = attention_dir + f'/{i}.pkl'
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
                    temporal_embeddings,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_tempo=use_tempo,
                    attention_fname=layer_attention_fname
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    temporal_embeddings,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    use_tempo=use_tempo,
                    attention_fname=layer_attention_fname
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
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def temporal_embedding_forward(self, temporal_embeddings, *args):
        return temporal_embeddings


class BertTemporalModel(BertModel):
    def __init__(self, config, add_pooling_layer=True, n_contexts=2):
        super().__init__(config, add_pooling_layer)

        self.encoder = BertTemporalEncoder(config)
        self.time_embeddings = nn.Embedding(2 + n_contexts, config.hidden_size)
        self.n_time_periods = n_contexts

        self.init_weights()
    
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
        use_tempo=True,
        attention_dir=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if timestamps is None:
            raise ValueError(f"You need to pass in a list of timestamps, ranging from 0 to {self.n_time_periods - 1}")
        elif timestamps.shape != input_ids.shape:
            raise ValueError(f'Timestamps not properly generated: must be same shape as input ids')

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if torch.max(input_ids) >= self.embeddings.word_embeddings.weight.shape[0]:
            raise ValueError(f"max of input ids is {torch.max(input_ids)} but emb len is {self.embeddings.word_embeddings.weight.shape[0]}!")
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        temporal_embeddings = self.time_embeddings(timestamps) 

        encoder_outputs = self.encoder(
            embedding_output,
            temporal_embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_tempo=use_tempo,
            attention_dir=attention_dir
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForTemporalMaskedLM(BertForMaskedLM):
    def __init__(self, config, n_contexts=2):
        super().__init__(config)
        self.bert = BertTemporalModel(config, n_contexts=n_contexts, add_pooling_layer=False)
        self.init_weights()

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
        use_tempo=True,
        attention_dir=None,
        **kwargs
    ):
        """
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
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            use_tempo=use_tempo,
            attention_dir=attention_dir
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )