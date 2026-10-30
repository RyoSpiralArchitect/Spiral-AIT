"""Curated multilingual corpora with licensing metadata for the demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class CorpusSample:
    """A single text sample paired with segmentation metadata."""

    language: str
    text: str
    segments: Tuple[str, ...]
    group: str


CORPUS_LICENSE: Dict[str, str] = {
    "id": "CC-BY-4.0",
    "name": "Creative Commons Attribution 4.0 International",
    "url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": "SpiralReality AIT authors",
    "notes": (
        "Synthetic reflective and multilingual narratives curated for the"
        " SpiralReality one-pass AIT demo. The texts were authored for this"
        " repository and may be redistributed under CC-BY-4.0."
    ),
}


_REFLECTIVE_RAW: Tuple[Tuple[str, str, str], ...] = (
    (
        "en",
        "Bob re-examined Carol's motives and updated his provisional evaluation.",
        "Bob|re|-|examined|Carol's|motives|and|updated|his|provisional|evaluation|.",
    ),
    (
        "en",
        "Avoid premature closure; maintain hypotheses and update them with evidence.",
        "Avoid|premature|closure|;|maintain|hypotheses|and|update|them|with|evidence|.",
    ),
    (
        "en",
        "Meta-analysis of sensor feeds prevents overfitting to the first plausible story.",
        "Meta|-|analysis|of|sensor|feeds|prevents|overfitting|to|the|first|plausible|story|.",
    ),
    (
        "en",
        "Iterative reframing of witness notes keeps latent clusters responsive to nuance.",
        "Iterative|reframing|of|witness|notes|keeps|latent|clusters|responsive|to|nuance|.",
    ),
    (
        "en",
        "Systematic journaling of counterfactuals stabilizes the agent's decision frontier.",
        "Systematic|journaling|of|counterfactuals|stabilizes|the|agent's|decision|frontier|.",
    ),
    (
        "en",
        "Layered attention summaries discourage impulsive segmentation during cold starts.",
        "Layered|attention|summaries|discourage|impulsive|segmentation|during|cold|starts|.",
    ),
    (
        "en",
        "Sustained reflective questioning keeps the analyst from freezing the latent plan too soon.",
        "Sustained|reflective|questioning|keeps|the|analyst|from|freezing|the|latent|plan|too|soon|.",
    ),
    (
        "en",
        "Calibrating priors against delayed confirmations avoids premature pruning of actions.",
        "Calibrating|priors|against|delayed|confirmations|avoids|premature|pruning|of|actions|.",
    ),
    (
        "en",
        "Contingency-aware sampling lets the bridge respect rare but critical micro-boundaries.",
        "Contingency|-|aware|sampling|lets|the|bridge|respect|rare|but|critical|micro|-|boundaries|.",
    ),
    (
        "en",
        "Slow breathing, careful parsing, then align evidence with provisional policies.",
        "Slow|breathing|,|careful|parsing|,|then|align|evidence|with|provisional|policies|.",
    ),
    (
        "en",
        "Maintain a rolling scratchpad; revise boundaries when fresh observations contradict assumptions.",
        "Maintain|a|rolling|scratchpad|;|revise|boundaries|when|fresh|observations|contradict|assumptions|.",
    ),
    (
        "en",
        "Socratic prompts elicit clarifications that re-open segments closed by cognitive fatigue.",
        "Socratic|prompts|elicit|clarifications|that|re|-|open|segments|closed|by|cognitive|fatigue|.",
    ),
    (
        "en",
        "Cross-lingual glossaries ensure the encoder does not conflate idioms with hard stops.",
        "Cross|-|lingual|glossaries|ensure|the|encoder|does|not|conflate|idioms|with|hard|stops|.",
    ),
    (
        "en",
        "Goal-conditioned rehearsal simulates failure cases before committing to gating decisions.",
        "Goal|-|conditioned|rehearsal|simulates|failure|cases|before|committing|to|gating|decisions|.",
    ),
    (
        "en",
        "Streaming diagnostics reveal when the CRF head saturates and needs regularization.",
        "Streaming|diagnostics|reveal|when|the|CRF|head|saturates|and|needs|regularization|.",
    ),
    (
        "en",
        "Noise-robust embeddings resist the temptation to snap to the majority class boundary.",
        "Noise|-|robust|embeddings|resist|the|temptation|to|snap|to|the|majority|class|boundary|.",
    ),
    (
        "en",
        "Phase-aware masking prevents the planner from extrapolating on stale latent states.",
        "Phase|-|aware|masking|prevents|the|planner|from|extrapolating|on|stale|latent|states|.",
    ),
    (
        "en",
        "Manual audits of auto-segmented transcripts surface drift before metrics collapse.",
        "Manual|audits|of|auto|-|segmented|transcripts|surface|drift|before|metrics|collapse|.",
    ),
    (
        "en",
        "Align observations, plans, and critiques; triangulated narratives yield resilient cuts.",
        "Align|observations|,|plans|,|and|critiques|;|triangulated|narratives|yield|resilient|cuts|.",
    ),
    (
        "en",
        "Async peer review catches overconfident merges in multilingual investigative reports.",
        "Async|peer|review|catches|overconfident|merges|in|multilingual|investigative|reports|.",
    ),
    (
        "en",
        "Temporal coherence checks highlight when the boundary signal lags the discourse flow.",
        "Temporal|coherence|checks|highlight|when|the|boundary|signal|lags|the|discourse|flow|.",
    ),
    (
        "ja",
        "ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。",
        "ボブ|は|キャロル|の|動機|を|再検討|し|、|第三者|の|証拠|で|暫定|評価|を|更新|した|。",
    ),
    (
        "ja",
        "結論を急ぎ過ぎないこと。内部対話で仮説を維持し、証拠で更新する。",
        "結論|を|急ぎ過ぎ|ない|こと|。|内部|対話|で|仮説|を|維持|し|、|証拠|で|更新|する|。",
    ),
)


_MULTILINGUAL_RAW: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "ja": (
        (
            "境界判定モデルは多言語入力でも一貫して学習される。",
            "境界|判定|モデル|は|多言語|入力|でも|一貫|して|学習|される|。",
        ),
        (
            "位相基底を共有すると変化点の感度が高まる。",
            "位相|基底|を|共有|すると|変化点|の|感度|が|高まる|。",
        ),
    ),
    "es": (
        (
            "La analista rastrea señales multilingües para ajustar los límites narrativos.",
            "La|analista|rastrea|señales|multilingües|para|ajustar|los|límites|narrativos|.",
        ),
        (
            "Modelos ligeros permiten entrenamiento estable sin depender de GPU.",
            "Modelos|ligeros|permiten|entrenamiento|estable|sin|depender|de|GPU|.",
        ),
    ),
    "fr": (
        (
            "Les frontières apprises restent fiables même lorsque la langue change.",
            "Les|frontières|apprises|restent|fiables|même|lorsque|la|langue|change|.",
        ),
        (
            "Un encodeur de phase affine les indices locaux pour guider la planification.",
            "Un|encodeur|de|phase|affine|les|indices|locaux|pour|guider|la|planification|.",
        ),
    ),
    "de": (
        (
            "Grenzsignale stabilisieren den Plan auch bei mehrsprachigen Dialogen.",
            "Grenzsignale|stabilisieren|den|Plan|auch|bei|mehrsprachigen|Dialogen|.",
        ),
        (
            "Ein lernbarer Phasenfilter verstärkt Übergänge mit geringer Latenz.",
            "Ein|lernbarer|Phasenfilter|verstärkt|Übergänge|mit|geringer|Latenz|.",
        ),
    ),
    "zh": (
        (
            "边界学生在多语言案例中保持高精度的分段预测。",
            "边界|学生|在|多语言|案例|中|保持|高精度|的|分段|预测|。",
        ),
        (
            "相位编码帮助注意力在跨域文本上快速适应。",
            "相位|编码|帮助|注意力|在|跨域|文本|上|快速|适应|。",
        ),
    ),
    "uk": (
        (
            "Модель читає український текст і перевіряє межі слів.",
            "Модель|читає|український|текст|і|перевіряє|межі|слів|.",
        ),
        (
            "Пам'ять зберігає попередні спостереження, а висновки змінюються.",
            "Пам'ять|зберігає|попередні|спостереження|,|а|висновки|змінюються|.",
        ),
        (
            "Об’єкт і зв’язок залишаються цілими словами.",
            "Об’єкт|і|зв’язок|залишаються|цілими|словами|.",
        ),
        (
            "Ґрунтовний аналіз допомагає перевірити гіпотези.",
            "Ґрунтовний|аналіз|допомагає|перевірити|гіпотези|.",
        ),
        (
            "Мо́ва і памʼять допомагають поєднувати нові спостереження.",
            "Мо́ва|і|памʼять|допомагають|поєднувати|нові|спостереження|.",
        ),
    ),
    "pl": (
        (
            "Model czyta polski tekst i sprawdza granice słów.",
            "Model|czyta|polski|tekst|i|sprawdza|granice|słów|.",
        ),
        (
            "Nowe obserwacje pomagają dokładniej ocenić wcześniejsze wnioski.",
            "Nowe|obserwacje|pomagają|dokładniej|ocenić|wcześniejsze|wnioski|.",
        ),
    ),
    "cs": (
        (
            "Model čte český text a kontroluje hranice slov.",
            "Model|čte|český|text|a|kontroluje|hranice|slov|.",
        ),
        (
            "Nová pozorování pomáhají zpřesnit předchozí závěry.",
            "Nová|pozorování|pomáhají|zpřesnit|předchozí|závěry|.",
        ),
    ),
}


def _make_sample(language: str, text: str, segments: str, group: str) -> CorpusSample:
    return CorpusSample(language=language, text=text, segments=tuple(segments.split("|")), group=group)


_REFLECTIVE_SAMPLES: Tuple[CorpusSample, ...] = tuple(
    _make_sample(lang, text, segments, "reflective")
    for lang, text, segments in _REFLECTIVE_RAW
)

_MULTILINGUAL_SAMPLES: Tuple[CorpusSample, ...] = tuple(
    _make_sample(lang, text, segments, "multilingual")
    for lang, items in _MULTILINGUAL_RAW.items()
    for text, segments in items
)

_ALL_SAMPLES: Tuple[CorpusSample, ...] = _REFLECTIVE_SAMPLES + _MULTILINGUAL_SAMPLES

REFLECTIVE_LANGUAGES: Tuple[str, ...] = tuple(sorted({sample.language for sample in _REFLECTIVE_SAMPLES}))


def iter_samples() -> Iterator[CorpusSample]:
    """Iterate over every curated sample."""

    return iter(_ALL_SAMPLES)


def reflective_samples() -> Tuple[CorpusSample, ...]:
    """Return the reflective base corpus used by the demo."""

    return _REFLECTIVE_SAMPLES


def samples_for_language(language: str) -> Tuple[CorpusSample, ...]:
    """Return all samples for ``language`` across both corpora."""

    return tuple(sample for sample in _ALL_SAMPLES if sample.language == language)


def languages() -> Tuple[str, ...]:
    """Languages available in the multilingual expansion corpora."""

    return tuple(sorted({sample.language for sample in _MULTILINGUAL_SAMPLES}))


def export_catalog(languages: Sequence[str] | None = None) -> Dict[str, object]:
    """Return a serialisable overview of the curated corpora."""

    if languages is None:
        languages = sorted({sample.language for sample in _ALL_SAMPLES})

    catalog: Dict[str, Dict[str, object]] = {}
    for code in languages:
        samples = samples_for_language(code)
        if not samples:
            continue
        groups = sorted({sample.group for sample in samples})
        examples = [sample.text for sample in samples[:2]]
        catalog[code] = {
            "count": len(samples),
            "groups": groups,
            "examples": examples,
        }

    return {
        "license": dict(CORPUS_LICENSE),
        "languages": catalog,
        "total_samples": sum(entry["count"] for entry in catalog.values()),
    }


__all__ = [
    "CORPUS_LICENSE",
    "CorpusSample",
    "REFLECTIVE_LANGUAGES",
    "export_catalog",
    "iter_samples",
    "languages",
    "reflective_samples",
    "samples_for_language",
]

