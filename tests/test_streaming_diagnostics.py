import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.streaming_inference import ChunkedStreamingSegmenter


def test_forced_split_count_survives_flush_and_explicit_reset_clears_it():
    stream = ChunkedStreamingSegmenter(lambda text: [text], max_window_chars=16, lookahead_chars=4)
    tokens = stream.feed("x" * 20)
    assert stream.forced_split_count == 1
    tokens += stream.flush()
    assert "".join(tokens) == "x" * 20
    assert stream.forced_split_count == 1
    stream.reset()
    assert stream.forced_split_count == 0


def test_failed_feed_rolls_back_forced_split_count_with_buffers():
    calls = []
    def segment(text):
        calls.append(text)
        if len(calls) == 2:
            raise ValueError("bad model result")
        return [text]
    stream = ChunkedStreamingSegmenter(segment, max_window_chars=16, lookahead_chars=4)
    with pytest.raises(ValueError):
        stream.feed("x" * 40)
    assert stream.pending_text == ""
    assert stream.forced_split_count == 0
