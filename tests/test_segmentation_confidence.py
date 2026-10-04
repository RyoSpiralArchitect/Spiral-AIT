from unittest.mock import patch

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT


@pytest.mark.parametrize("aif", [False, True])
@pytest.mark.parametrize("text", ["", "猫", "Check each boundary. 境界を確認。"])
def test_confidence_and_spans_share_decoded_model(text, aif):
    model = OnePassAIT(latent_dim=16, encoder_layers=1)
    plain = model.segment_text(text, use_aif=aif)
    with patch.object(model.encoder, "forward", wraps=model.encoder.forward) as forward:
        result = model.segment_text(text, use_aif=aif, include_confidence=True)
        assert forward.call_count == (1 if len(text) > 1 else 0)
    assert result["tokens"] == plain
    assert len(result["boundary_probabilities"]) == max(0, len(text) - 1)
    assert 0 <= result["boundary_entropy"] <= np.log(2)
    expected = model.student.boundary_probs_with_logit_bias(text, result["logit_bias"])
    np.testing.assert_allclose(result["boundary_probabilities"], expected)
    assert len(result["tokens"]) == len(result["spans"])
    for token, span in zip(result["tokens"], result["spans"]):
        assert text[span["start"]:span["end"]] == token
        if span["end"] < len(text):
            assert span["boundary_probability"] == expected[span["end"] - 1]
        else:
            assert span["boundary_probability"] is None


def test_confident_stream_uses_model_metadata_and_preserves_text():
    model = OnePassAIT(latent_dim=16, encoder_layers=1)
    stream = model.streaming_segmenter(
        max_window_chars=32, lookahead_chars=8, context_chars=8,
        min_boundary_confidence=0.99, use_aif=True,
    )
    text = "Uncertain boundaries can wait for more context. 猫も待つ。"
    out = []
    for ch in text:
        out.extend(stream.feed(ch))
        assert len(stream.pending_text) <= 32
    out.extend(stream.flush())
    assert "".join(out) == text
    assert not stream.pending_text
