import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from generation_trie import Trie


class ImageAssistedT5(T5ForConditionalGeneration):
    """Use the text decoder state to read the user's image-code history."""

    def __init__(self, config):
        super().__init__(config)
        self.image_attention = nn.MultiheadAttention(
            config.d_model, 1, dropout=config.dropout_rate, batch_first=True
        )
        self.candidate_trie = None

    def set_candidate_items(self, tokenizer, items):
        self.candidate_trie = Trie([
            [self.config.decoder_start_token_id] + tokenizer.encode(item)
            for item in sorted(items)
        ])
        return self.candidate_trie

    def freeze_text_backbone(self):
        self.requires_grad_(False)
        self.image_attention.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not self.shared.weight.requires_grad:
            self.encoder.eval()
            self.decoder.eval()
        return self

    def encode_image_history(self, image_input_ids, image_attention_mask):
        return self.encoder(
            input_ids=image_input_ids, attention_mask=image_attention_mask, return_dict=True
        ).last_hidden_state

    def entropy_weights(self, logits, prefixes):
        # Cached decoding returns only the last position. Teacher forcing returns all positions.
        start = prefixes.size(1) - logits.size(1)
        successors = [
            self.candidate_trie.get(prefix[:end])
            for prefix in prefixes.tolist()
            for end in range(start + 1, len(prefix) + 1)
        ]
        width = max(1, max(map(len, successors)))
        indices = torch.tensor(
            [tokens + [-1] * (width - len(tokens)) for tokens in successors],
            device=logits.device,
        )
        valid = indices >= 0
        counts = valid.sum(dim=-1)
        # A dummy slot keeps the softmax finite for completed or padded prefixes.
        valid[:, 0] = True
        scores = logits.detach().flatten(0, 1).gather(1, indices.clamp_min(0)).float()
        log_probs = scores.masked_fill(~valid, -torch.inf).log_softmax(dim=-1)
        entropy = -(log_probs.exp() * log_probs.masked_fill(~valid, 0)).sum(dim=-1)
        weights = entropy / counts.clamp_min(2).float().log()
        return weights.clamp(0, 1).reshape(*logits.shape[:2], 1).to(logits.dtype)

    @torch.no_grad()
    def generate(self, *args, image_input_ids=None, image_attention_mask=None, **kwargs):
        if image_input_ids is not None:
            kwargs["image_memory"] = self.encode_image_history(image_input_ids, image_attention_mask)
            kwargs["image_attention_mask"] = image_attention_mask
        return super().generate(*args, **kwargs)

    def prepare_inputs_for_generation(
        self, input_ids, image_memory=None, image_attention_mask=None, **kwargs
    ):
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if image_memory is not None:
            inputs.update(
                image_memory=image_memory,
                image_attention_mask=image_attention_mask,
                full_decoder_prefix=input_ids,
            )
        return inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        image_input_ids=None,
        image_attention_mask=None,
        image_memory=None,
        full_decoder_prefix=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        base_inputs = dict(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if image_input_ids is None and image_memory is None:
            return super().forward(
                **base_inputs, decoder_input_ids=decoder_input_ids, labels=labels,
                output_hidden_states=output_hidden_states, return_dict=return_dict,
            )

        gate = self.config.image_history_gate
        if gate == "entropy" and self.candidate_trie is None:
            raise ValueError("Call set_candidate_items before using the entropy gate.")
        if image_memory is None:
            image_memory = self.encode_image_history(image_input_ids, image_attention_mask)
        if labels is not None:
            base_inputs.setdefault("use_cache", False)
            if decoder_input_ids is None:
                decoder_input_ids = self._shift_right(labels)

        outputs = super().forward(
            **base_inputs, decoder_input_ids=decoder_input_ids,
            output_hidden_states=True, return_dict=True,
        )
        hidden = outputs.decoder_hidden_states[-1]
        if gate == "entropy":
            prefix = decoder_input_ids if full_decoder_prefix is None else full_decoder_prefix
            weight = self.entropy_weights(outputs.logits, prefix)
        else:
            weight = 1.0

        context, _ = self.image_attention(
            hidden, image_memory, image_memory,
            key_padding_mask=image_attention_mask.eq(0) if image_attention_mask is not None else None,
            need_weights=False,
        )
        residual = weight * context
        if self.config.tie_word_embeddings:
            residual = residual / math.sqrt(self.model_dim)
        # The T5 LM head is linear, so this is LMHead(hidden + weight * context).
        outputs.logits = outputs.logits + self.lm_head(residual)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        expose_hidden = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        result = Seq2SeqLMOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states if expose_hidden else None,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states if expose_hidden else None,
            encoder_attentions=outputs.encoder_attentions,
        )
        as_dict = self.config.use_return_dict if return_dict is None else return_dict
        return result if as_dict else result.to_tuple()
