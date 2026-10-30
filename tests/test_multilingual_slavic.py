import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.multilingual import (
    AVAILABLE_LANGUAGES, SLAVIC_LANGUAGES, build_multilingual_corpus,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT, StudentTrainingConfig


def test_slavic_demo_pack_preserves_text_apostrophes_and_combining_stress():
    assert set(SLAVIC_LANGUAGES) <= set(AVAILABLE_LANGUAGES)
    texts, labels, tags = build_multilingual_corpus(SLAVIC_LANGUAGES, include_reflective=False, shuffle=False)
    assert set(tags) == {"uk", "pl", "cs"}
    assert all("".join(parts) == text for text, parts in zip(texts, labels))
    ukrainian = [text for text, language in zip(texts, tags) if language == "uk"]
    assert all(any(character in text for text in ukrainian) for character in ("Ґ", "є", "ї", "'", "’", "ʼ", "\u0301"))
    assert any("Мо\u0301ва" in text for text in ukrainian)


@pytest.mark.parametrize("language", SLAVIC_LANGUAGES)
def test_slavic_language_training_api_and_checkpoint_reconstruct_authored_text(language):
    model = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy", seed=19)
    summary = model.train_student(languages=(language,), include_reflective=False,
                                  cfg=StudentTrainingConfig(epochs=1, validation_split=0,
                                                            phase_lr=0, lexical_buckets=32))
    assert summary["dataset_languages"] == {language: summary["dataset_size"]}
    texts, _, _ = build_multilingual_corpus((language,), include_reflective=False, shuffle=False)
    restored = OnePassAIT(latent_dim=4, encoder_layers=1, encoder_heads=2, encoder_backend="numpy")
    restored.load_state_dict(model.state_dict())
    for text in texts:
        before = model.segment_text(text, include_confidence=True)
        assert "".join(before["tokens"]) == text
        assert before == restored.segment_text(text, include_confidence=True)
