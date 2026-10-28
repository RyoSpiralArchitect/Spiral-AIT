from dataclasses import replace
import json

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import (
    ExternalSentence, LANGUAGES, group_parallel_corpus, load_pinned_sources, parse_conllu, sha256,
)


def row(token_id, form):
    return "\t".join([token_id, form] + ["_"] * 8)


def test_conllu_surface_tokens_replace_mwt_components_and_preserve_whitespace():
    content = "\n".join([
        "# newdoc id = document", "# sent_id = sentence", "# parallel_id = pud/sentence",
        "# text = I can't  go.", row("1", "I"), row("2-3", "can't"),
        row("2", "ca"), row("3", "n't"), row("3.1", "phantom"), row("4", "go"), row("5", "."),
    ])
    records, excluded = parse_conllu(content, "en")
    assert excluded == []
    assert records[0].segments == ("I", " ", "can't", "  ", "go", ".")
    assert "".join(records[0].segments) == records[0].text


def test_conllu_unalignable_forms_are_reported_instead_of_normalized():
    source = "\n".join(["# newdoc id = d", "# sent_id = s", "# parallel_id = pud/s", "# text = Ａ", row("1", "A")])
    records, excluded = parse_conllu(source, "ja")
    assert records == []
    assert excluded == [{"language": "ja", "sentence_id": "s", "reason": "surface_form_alignment"}]
    with pytest.raises(ValueError, match="Explicit document"):
        parse_conllu(source.replace("# newdoc id = d\n", ""), "ja")


def corpus_rows():
    return [ExternalSentence(lang, f"d{index:02d}", f"s{index:02d}", f"pud/s{index:02d}",
                             f"{lang}-{index}", (lang, "-", str(index)))
            for index in range(16) for lang in LANGUAGES]


def test_document_and_translation_groups_never_cross_train_dev_test():
    rows = corpus_rows()
    for lang in LANGUAGES:
        rows.append(ExternalSentence(lang, "d00", "another", "pud/another", f"another-{lang}", (f"another-{lang}",)))
    grouped = group_parallel_corpus(rows, folds=4)
    assert grouped.receipt == group_parallel_corpus(list(reversed(rows)), folds=4).receipt
    all_test_ids = []
    for fold in range(4):
        partitions, receipt = grouped.partition(fold, max_train_groups=2)
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            assert set(receipt[left]["documents"]).isdisjoint(receipt[right]["documents"])
            assert set(receipt[left]["parallel_ids"]).isdisjoint(receipt[right]["parallel_ids"])
        for name in partitions:
            counts = receipt[name]["per_language"]
            assert counts["en"] == counts["ja"] == counts["zh"]
        all_test_ids.extend(receipt["test"]["parallel_ids"])
    assert len(all_test_ids) == len(set(all_test_ids)) == 17


def test_normalized_duplicates_union_documents_and_conflicting_labels_fail():
    rows = corpus_rows()
    rows[3] = replace(rows[3], text=rows[0].text.upper(), segments=tuple(piece.upper() for piece in rows[0].segments))
    grouped = group_parallel_corpus(rows, folds=4)
    assert grouped.group_for_document["d00"] == grouped.group_for_document["d01"]
    rows[3] = replace(rows[3], text=rows[0].text, segments=(rows[0].text,))
    with pytest.raises(ValueError, match="conflicting supervision"):
        group_parallel_corpus(rows, folds=4)


def test_length_or_missing_language_excludes_the_entire_parallel_sentence():
    rows = corpus_rows()[1:]
    rows[5] = replace(rows[5], text="x" * 40, segments=("x" * 40,))
    grouped = group_parallel_corpus(rows, folds=4, max_chars=20)
    assert len(grouped.receipt["exclusions"]) == 2
    assert len(grouped.records) == 42


def test_pinned_sources_require_matching_local_checksums(tmp_path):
    sources = []
    for lang in LANGUAGES:
        revision = "a" * 40
        directory = tmp_path / "cache" / lang / revision
        directory.mkdir(parents=True)
        content = "\n".join(["# newdoc id = d", "# sent_id = s", "# parallel_id = pud/s", "# text = ab", row("1", "ab")])
        files = {f"{lang}_pud-ud-test.conllu": content, "README.md": "attribution", "LICENSE.txt": "license"}
        metadata = {}
        for name, text in files.items():
            payload = text.encode()
            (directory / name).write_bytes(payload)
            metadata[name] = {"bytes": len(payload), "sha256": sha256(payload), "url": "unused-offline"}
        sources.append({"language": lang, "revision": revision, "files": metadata})
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"sources": sources}))
    records, _ = load_pinned_sources(manifest, tmp_path / "cache")
    assert len(records) == 3
    (tmp_path / "cache" / "ja" / ("a" * 40) / "ja_pud-ud-test.conllu").write_text("corrupt")
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_pinned_sources(manifest, tmp_path / "cache")
