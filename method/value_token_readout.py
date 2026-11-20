"""Inference adapter for VGAS Q_Former heads which read outputs[:, -1, :]."""

from torch import nn


class ValueTokenDecoder(nn.Module):
    """Return only the terminal value representation of a frozen Q_Former.

    Earlier layers and final attention see the complete sequence. After that
    attention, all remaining operations act independently on each token.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, hidden_states, attention_mask, position_ids):
        if self.backbone.training:
            raise RuntimeError("Value-token readout requires an evaluation-mode critic")
        backbone = self.backbone
        kwargs = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=backbone.rotary_emb(hidden_states, position_ids),
            past_key_values=None,
            cache_position=None,
            output_attentions=False,
            use_cache=False,
        )
        for layer in backbone.layers[:-1]:
            output = layer(hidden_states, **kwargs)
            hidden_states = output[0] if isinstance(output, tuple) else output
        last = backbone.layers[-1]
        attention, _ = last.self_attn(last.input_layernorm(hidden_states), **kwargs)
        value = hidden_states[:, -1:] + attention[:, -1:]
        return backbone.norm(value + last.mlp(last.post_attention_layernorm(value)))


def value_token_decoders(critic):
    """Build optional replacements for the pinned VGAS twin heads after loading."""
    from qchunk.valuequeryhead import Q_Former, Q_Former_Backbone

    heads = [branch.head for branch in critic.heads]
    if critic.training or not all(
        type(head) is Q_Former and type(head.decoder) is Q_Former_Backbone
        for head in heads
    ):
        raise ValueError("Expected frozen, evaluation-mode VGAS Q_Former heads")
    return [ValueTokenDecoder(head.decoder).eval() for head in heads]
