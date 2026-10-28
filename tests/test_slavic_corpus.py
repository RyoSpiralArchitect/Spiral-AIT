from dataclasses import replace
import json
from pathlib import Path

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.external_corpus import normalized_text, sha256
from spiralreality_AIT_onepass_aifcore_integrated.integrated import slavic_corpus
from spiralreality_AIT_onepass_aifcore_integrated.integrated.slavic_corpus import (
    SlavicSentence, load_slavic_sources, parse_slavic_conllu, partition_ukrainian,
)


def token(token_id, form):
    return "\t".join([token_id, form] + ["_"] * 8)


def native_block(sentence_id, text, forms=None, document=None):
    lines = [f"# sent_id = {sentence_id}", f"# text = {text}"]
    if document:
        lines.insert(0, f"# newdoc_id = {document}")
    lines.extend(token(str(index), form) for index, form in enumerate(forms or [text], 1))
    return "\n".join(lines)


def parse_native(content):
    return parse_slavic_conllu(content, "uk", source_id="UD_Ukrainian-ParlaMint", upstream_split="train")


def test_native_preserves_apostrophes_combining_marks_and_unicode_whitespace_exactly():
    words = ["п'ять", "п’ять", "пʼять", "ї\u0301жа", "ґанок"]
    text = "\u00a0" + "\u2028".join(words) + "  "
    content = native_block("ParlaMint-UA_2022-01-25-m0.u1.p1.lang1.s1", text, words,
                           "ParlaMint-UA_2022-01-25-m0.u1")
    rows, exclusions = parse_native(content)
    assert exclusions == []
    assert rows[0].text == text
    assert "".join(rows[0].segments) == text
    assert rows[0].parallel_id is None
    assert rows[0].document_id == "ParlaMint-UA_2022-01-25"
    assert rows[0].segments[0] == "\u00a0" and rows[0].segments[-1] == "  "
    # Both upstream metadata spellings are supported without altering text.
    assert parse_native(content.replace("newdoc_id", "newdoc id"))[0] == rows


def test_native_mid_document_source_and_documented_nsdc_transcript_have_real_groups():
    blocks = [native_block("ParlaMint-UA_2022-01-25-m0.u1.p1.lang1.s1", "Один"),
              native_block("ParlaMint-UA_2022-01-25-m1.u2.p1.lang1.s1", "Два"),
              native_block("NSDC_UA_28_Feb2014-1", "Три")]
    rows, _ = parse_native("\n\n".join(blocks))
    assert rows[0].document_id == rows[1].document_id == "ParlaMint-UA_2022-01-25"
    assert rows[2].document_id == "NSDC_UA_28_Feb2014"
    with pytest.raises(ValueError, match="documented native"):
        parse_native(native_block("unknown-sentence-1", "Один"))
    with pytest.raises(ValueError, match="disagrees"):
        parse_native(native_block("ParlaMint-UA_2022-01-25-m0.u1.p1.lang1.s1", "Один",
                                  document="ParlaMint-UA_2022-02-24-m0.u1"))


def test_surface_mwt_replaces_components_and_empty_node_zero_is_ignored():
    content = "\n".join(["# sent_id = pud-s", "# parallel_id = pud/p", "# text = aby  šel.",
                         token("0.1", "phantom"), token("1-2", "aby"), token("1", "a"),
                         token("2", "by"), token("2.1", "phantom"), token("3", "šel"), token("4", ".")])
    rows, exclusions = parse_slavic_conllu(content, "cs", source_id="UD_Czech-PUD",
                                           upstream_split="test", pud_document_for_parallel={"pud/p": "real-doc"})
    assert exclusions == []
    assert rows[0].segments == ("aby", "  ", "šel", ".")
    assert rows[0].document_id == "real-doc"


def test_pud_refuses_missing_or_conflicting_original_document_metadata():
    content = "\n".join(["# sent_id = sentence", "# parallel_id = pud/p", "# text = Kot", token("1", "Kot")])
    kwargs = {"source_id": "UD_Polish-PUD", "upstream_split": "test"}
    with pytest.raises(ValueError, match="requires original"):
        parse_slavic_conllu(content, "pl", **kwargs)
    with pytest.raises(ValueError, match="lacks original"):
        parse_slavic_conllu(content, "pl", pud_document_for_parallel={}, **kwargs)
    with pytest.raises(ValueError, match="disagrees"):
        parse_slavic_conllu("# newdoc id = wrong\n" + content, "pl",
                            pud_document_for_parallel={"pud/p": "real-doc"}, **kwargs)


def test_source_alignment_is_never_normalized_or_repaired():
    content = native_block("ParlaMint-UA_2022-01-25-m0.u1.p1.lang1.s1", "й", ["и\u0306"])
    rows, exclusions = parse_native(content)
    assert rows == []
    assert exclusions[0]["reason"] == "surface_form_alignment"
    assert exclusions[0]["text_sha256"] == sha256("й".encode())
    assert "text" not in exclusions[0]
    with pytest.raises(ValueError, match="missing component"):
        parse_native(content.rsplit("\n", 1)[0] + "\n" + token("1-2", "й") + "\n" + token("1", "й"))


def native_rows():
    rows = []
    for index in range(6):
        document = f"ParlaMint-UA_2022-01-{index + 10:02d}"
        for sentence_index in range(2):
            text = f"Слово {index} {sentence_index}"
            rows.append(SlavicSentence("uk", document, f"{document}-m0.u1.p1.lang1.s{sentence_index + 1}",
                                       text, (text,), ("train", "dev", "test")[index % 3],
                                       "UD_Ukrainian-ParlaMint"))
    return rows


def test_native_dates_are_disjoint_and_partition_does_not_use_upstream_roles():
    rows = native_rows()
    partitions, receipt = partition_ukrainian(rows)
    reverse_partitions, reverse_receipt = partition_ukrainian(list(reversed(rows)))
    assert partitions == reverse_partitions and receipt == reverse_receipt
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert set(receipt[left]["documents"]).isdisjoint(receipt[right]["documents"])
    changed_roles = [replace(row, upstream_split="train") for row in rows]
    changed, _ = partition_ukrainian(changed_roles)
    assert {role: [row.sentence_id for row in values] for role, values in partitions.items()} == {
        role: [row.sentence_id for row in values] for role, values in changed.items()}
    assert receipt["folds"] == 5


def test_common_parliamentary_phrase_is_excluded_across_roles_without_collapsing_dates():
    rows = native_rows()
    _, initial = partition_ukrainian(rows)
    selected = {}
    for row in rows:
        selected.setdefault(initial["role_for_document"][row.document_id], row)
    for index, row in enumerate(selected.values()):
        rows.append(replace(row, sentence_id=row.sentence_id.rsplit("s", 1)[0] + f"s{index + 20}",
                            text="Дякую.", segments=("Дякую", ".")))
    partitions, receipt = partition_ukrainian(rows)
    assert receipt["document_groups"] == 6
    assert all(row.text != "Дякую." for values in partitions.values() for row in values)
    cross = [row for row in receipt["exclusions"] if row["reason"] == "normalized_duplicate_cross_role"]
    assert len(cross) == 3 and {row["role"] for row in cross} == {"train", "dev", "test"}
    assert all("text" not in row and "normalized_text_sha256" in row for row in cross)


def test_within_role_duplicates_keep_one_and_length_filter_follows_global_dedup():
    rows = native_rows()
    source = rows[0]
    rows.append(replace(source, sentence_id=source.sentence_id.rsplit("s", 1)[0] + "s30"))
    _, receipt = partition_ukrainian(rows)
    assert sum(row["reason"] == "normalized_duplicate_within_role" for row in receipt["exclusions"]) == 1
    for left in ("train", "dev", "test"):
        partitions, _ = partition_ukrainian(rows)
        for right in ("train", "dev", "test"):
            if left != right:
                assert {normalized_text(row.text) for row in partitions[left]}.isdisjoint(
                    normalized_text(row.text) for row in partitions[right])
    with pytest.raises(ValueError, match="nonempty"):
        partition_ukrainian(rows, max_chars=2)


def test_native_partition_refuses_insufficient_real_dates_and_invalid_labels():
    with pytest.raises(ValueError, match="independent"):
        partition_ukrainian(native_rows()[:4])
    partitions, receipt = partition_ukrainian(native_rows()[:6], folds=10)
    assert receipt["folds"] == 3 and all(partitions.values())
    for invalid in (True, 2, 3.0):
        with pytest.raises(ValueError, match="folds"):
            partition_ukrainian(native_rows(), folds=invalid)
    rows = native_rows()
    rows[0] = replace(rows[0], segments=("wrong",))
    with pytest.raises(ValueError, match="reconstruct"):
        partition_ukrainian(rows)


def fixture_manifest(tmp_path):
    manifest = json.loads((Path(__file__).parents[1] / "configs/slavic-v2.18-sources.json").read_text())
    cache = tmp_path / "cache"
    for source in manifest["sources"]:
        for filename, item in source["files"].items():
            if filename.endswith(".conllu"):
                if source["language"] == "uk":
                    payload = native_block(f"ParlaMint-UA_2022-01-25-m0.u1.p1.lang1.s{['train','dev','test'].index(item['upstream_split'])+1}", "Текст").encode()
                else:
                    payload = "\n".join(["# sent_id = s", "# parallel_id = pud/p", "# text = Text", token("1", "Text")]).encode()
            else:
                payload = ("license" if filename == "LICENSE.txt" else "attribution").encode()
            item.update(bytes=len(payload), sha256=sha256(payload))
            target = cache / source["language"] / source["revision"] / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return manifest, path, cache


def test_offline_source_bytes_and_checksums_are_verified_before_parsing(tmp_path, monkeypatch):
    manifest, path, cache = fixture_manifest(tmp_path)
    monkeypatch.setattr(slavic_corpus, "urlopen", lambda *args, **kwargs: pytest.fail("Unexpected network access"))
    rows, receipt = load_slavic_sources(path, cache, pud_document_for_parallel={"pud/p": "doc"})
    assert len(rows) == 5
    assert receipt["alignment_exclusions"] == []
    assert receipt["source_manifest_sha256"] == sha256(path.read_bytes())
    source = manifest["sources"][0]
    target = cache / "uk" / source["revision"] / "LICENSE.txt"
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_slavic_sources(path, cache, pud_document_for_parallel={"pud/p": "doc"})


@pytest.mark.parametrize("mutation", ["url", "size", "digest", "revision", "license", "extra_language"])
def test_manifest_rejects_unpinned_or_unbounded_sources_before_network(tmp_path, monkeypatch, mutation):
    manifest, path, cache = fixture_manifest(tmp_path)
    source = manifest["sources"][0]
    first = next(iter(source["files"].values()))
    if mutation == "url": first["url"] = "https://example.com/unpinned"
    elif mutation == "size": first["bytes"] = 32 * 1024 * 1024 + 1
    elif mutation == "digest": first["sha256"] = "not-a-digest"
    elif mutation == "revision": source["revision"] = "main"
    elif mutation == "license": source["license"] = "unknown"
    else: manifest["sources"].append(source)
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(slavic_corpus, "urlopen", lambda *args, **kwargs: pytest.fail("Unexpected network access"))
    with pytest.raises(ValueError):
        load_slavic_sources(path, cache, pud_document_for_parallel={"pud/p": "doc"}, download=True)


def test_download_requires_opt_in_and_never_retries_failed_request(tmp_path, monkeypatch):
    manifest, path, cache = fixture_manifest(tmp_path)
    source = manifest["sources"][0]
    target = cache / "uk" / source["revision"] / next(iter(source["files"]))
    target.unlink()
    calls = []
    def fail_once(url, **kwargs):
        calls.append(url)
        raise OSError("one failed request")
    monkeypatch.setattr(slavic_corpus, "urlopen", fail_once)
    with pytest.raises(FileNotFoundError, match="explicit"):
        load_slavic_sources(path, cache, pud_document_for_parallel={"pud/p": "doc"})
    assert calls == []
    with pytest.raises(OSError, match="one failed"):
        load_slavic_sources(path, cache, pud_document_for_parallel={"pud/p": "doc"}, download=True)
    assert len(calls) == 1 and not target.exists()
