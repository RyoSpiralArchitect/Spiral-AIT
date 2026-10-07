"""Pinned CoNLL-U input and document/translation-safe evaluation partitions.

No download occurs on import. Raw text is kept in a caller-selected local cache;
receipts contain provenance and hashes, not republished corpus text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import urlopen


LANGUAGES = ("en", "ja", "zh")
MAX_SOURCE_BYTES = 32 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class ExternalSentence:
    language: str
    document_id: str
    sentence_id: str
    parallel_id: str
    text: str
    segments: tuple[str, ...]


def parse_conllu(content: str, language: str):
    """Use surface MWT rows, ignore empty nodes, and preserve exact whitespace.

    Missing provenance or malformed IDs are errors. Unalignable source sentences
    are explicitly excluded; token normalization or invented labels are forbidden.
    """
    records, excluded = [], []
    document_id = None
    seen = set()
    for block in re.split(r"\n\s*\n", content.strip()):
        metadata, rows = {}, []
        for line in block.splitlines():
            if line.startswith("#"):
                key, separator, value = line[2:].partition(" = ")
                if separator:
                    metadata[key] = value
            elif line:
                fields = line.split("\t")
                if len(fields) != 10:
                    raise ValueError("CoNLL-U token rows must have ten fields")
                rows.append(fields)
        if "newdoc id" in metadata:
            document_id = metadata["newdoc id"]
        if not rows:
            continue
        required = ("sent_id", "parallel_id", "text")
        if not document_id or any(not metadata.get(key) for key in required):
            raise ValueError("Explicit document, sentence, parallel, and text metadata are required")
        sentence_id = metadata["sent_id"]
        if sentence_id in seen:
            raise ValueError("Duplicate sentence ID")
        seen.add(sentence_id)
        forms, covered_until, previous_id = [], 0, 0
        for fields in rows:
            token_id, form = fields[:2]
            if re.fullmatch(r"[1-9]\d*\.[1-9]\d*", token_id):
                continue
            if re.fullmatch(r"[1-9]\d*-[1-9]\d*", token_id):
                start, end = map(int, token_id.split("-"))
                if start != previous_id + 1 or start <= covered_until or end <= start:
                    raise ValueError("Invalid or overlapping multiword-token range")
                forms.append(form)
                covered_until = end
                continue
            if not re.fullmatch(r"[1-9]\d*", token_id) or int(token_id) != previous_id + 1:
                raise ValueError("Token IDs must be consecutive integers")
            previous_id = int(token_id)
            if previous_id > covered_until:
                forms.append(form)
        if covered_until > previous_id:
            raise ValueError("Multiword-token range has missing component rows")
        text = metadata["text"]
        segments, cursor = [], 0
        for form in forms:
            start = text.find(form, cursor) if form else -1
            gap = text[cursor:start] if start >= 0 else ""
            if start < 0 or (gap and not gap.isspace()):
                break
            if gap:
                segments.append(gap)
            segments.append(form)
            cursor = start + len(form)
        else:
            suffix = text[cursor:]
            if not suffix or suffix.isspace():
                if suffix:
                    segments.append(suffix)
                if "".join(segments) == text:
                    records.append(ExternalSentence(language, document_id, sentence_id,
                                                     metadata["parallel_id"], text, tuple(segments)))
                    continue
        excluded.append({"language": language, "sentence_id": sentence_id, "reason": "surface_form_alignment"})
    return records, excluded


def load_pinned_sources(manifest_path, cache_dir, *, download=False):
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    records, excluded, receipts = [], [], []
    languages = [source["language"] for source in manifest["sources"]]
    if sorted(languages) != sorted(LANGUAGES):
        raise ValueError("Source manifest must contain exactly English, Japanese, and Chinese")
    for source in manifest["sources"]:
        language, revision = source["language"], source["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Sources must pin an immutable Git revision")
        contents = {}
        for filename, metadata in source["files"].items():
            if Path(filename).name != filename or "\\" in filename:
                raise ValueError("Source filenames must be basenames")
            if not 0 < metadata["bytes"] <= MAX_SOURCE_BYTES:
                raise ValueError("Source byte length exceeds the acquisition bound")
            destination = Path(cache_dir) / language / revision / filename
            if not destination.exists():
                if not download:
                    raise FileNotFoundError(f"Missing source {filename}; use the explicit --download option")
                if not metadata["url"].startswith("https://raw.githubusercontent.com/UniversalDependencies/") or f"/{revision}/" not in metadata["url"]:
                    raise ValueError("Download URL must reference the pinned upstream revision")
                with urlopen(metadata["url"], timeout=45) as response:
                    payload = response.read(metadata["bytes"] + 1)
                if len(payload) != metadata["bytes"] or sha256(payload) != metadata["sha256"]:
                    raise ValueError(f"Source integrity mismatch: {filename}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            payload = destination.read_bytes()
            if len(payload) != metadata["bytes"] or sha256(payload) != metadata["sha256"]:
                raise ValueError(f"Source integrity mismatch: {filename}")
            contents[filename] = payload.decode("utf-8")
        if not {"README.md", "LICENSE.txt", f"{language}_pud-ud-test.conllu"} <= contents.keys():
            raise ValueError("Source manifest must pin data, attribution, and license files")
        parsed, rejected = parse_conllu(contents[f"{language}_pud-ud-test.conllu"], language)
        records.extend(parsed)
        excluded.extend(rejected)
        receipts.append({**source, "aligned_sentences": len(parsed), "unalignable_sentences": len(rejected)})
    return records, {"source_manifest_sha256": sha256(manifest_bytes), "sources": receipts,
                     "alignment_exclusions": excluded,
                     "tokenization": "UD surface tokens (MWT rows replace components); whitespace gaps are separate exact spans"}


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass
class GroupedCorpus:
    records: tuple[ExternalSentence, ...]
    group_for_document: dict[str, str]
    fold_for_group: dict[str, int]
    receipt: dict

    def partition(self, test_fold: int, *, max_train_groups=None):
        folds = self.receipt["folds"]
        if isinstance(test_fold, bool) or not 0 <= test_fold < folds:
            raise ValueError("test fold is outside the frozen partition")
        if max_train_groups is not None and (isinstance(max_train_groups, bool) or not isinstance(max_train_groups, int) or max_train_groups < 1):
            raise ValueError("max_train_groups must be a positive integer")
        dev_fold = (test_fold + 1) % folds
        groups = {name: [] for name in ("train", "dev", "test")}
        for group in self.fold_for_group:  # frozen hash order
            fold = self.fold_for_group[group]
            role = "test" if fold == test_fold else "dev" if fold == dev_fold else "train"
            groups[role].append(group)
        if max_train_groups is not None:
            groups["train"] = groups["train"][:max_train_groups]
        partitions = {name: tuple(row for row in self.records if self.group_for_document[row.document_id] in set(selected))
                      for name, selected in groups.items()}
        if any(not rows for rows in partitions.values()):
            raise ValueError("All train/dev/test partitions must be nonempty")
        receipt = {"test_fold": test_fold, "dev_fold": dev_fold, "max_train_groups": max_train_groups}
        for name, rows in partitions.items():
            receipt[name] = {"groups": groups[name], "documents": sorted({row.document_id for row in rows}),
                             "parallel_ids": sorted({row.parallel_id for row in rows}),
                             "samples": len(rows), "per_language": {lang: sum(row.language == lang for row in rows) for lang in LANGUAGES},
                             "labels_sha256": json_hash([asdict(row) for row in rows])}
        return partitions, receipt


def group_parallel_corpus(records, *, seed=1739, folds=10, max_chars=256):
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 3:
        raise ValueError("At least three integer folds are required")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 2:
        raise ValueError("max_chars must be an integer of at least two")
    parallel, unique_text = {}, {}
    for row in records:
        if row.language not in LANGUAGES or not row.document_id or not row.parallel_id:
            raise ValueError("Missing grouping provenance")
        if not row.segments or any(not part for part in row.segments) or "".join(row.segments) != row.text:
            raise ValueError("Labels must reconstruct the source text exactly")
        key = (row.language, row.text)
        if key in unique_text and unique_text[key] != row.segments:
            raise ValueError("Identical source text has conflicting supervision")
        unique_text[key] = row.segments
        group = parallel.setdefault(row.parallel_id, {})
        if row.language in group:
            raise ValueError("Duplicate language/parallel ID")
        group[row.language] = row
    eligible, excluded = [], []
    for parallel_id, by_language in sorted(parallel.items()):
        reason = None
        if set(by_language) != set(LANGUAGES):
            reason = "missing_aligned_language"
        elif len({row.document_id for row in by_language.values()}) != 1:
            raise ValueError("Parallel sentences disagree about their source document")
        elif any(not 2 <= len(row.text) <= max_chars for row in by_language.values()):
            reason = "parallel_group_length_filter"
        if reason:
            excluded.append({"parallel_id": parallel_id, "reason": reason})
        else:
            eligible.extend(by_language[lang] for lang in LANGUAGES)
    # Union every document containing an identical or formatting-equivalent
    # string, even when that sentence is excluded by the common-length filter.
    # This deliberately prefers conservative over-grouping to leakage.
    parents = {row.document_id: row.document_id for row in records}
    def root(doc):
        while parents[doc] != doc:
            parents[doc] = parents[parents[doc]]
            doc = parents[doc]
        return doc
    text_owner = {}
    for row in records:
        key = normalized_text(row.text)
        if key in text_owner:
            left, right = sorted((root(row.document_id), root(text_owner[key])))
            parents[right] = left
        else:
            text_owner[key] = row.document_id
    group_for_document = {doc: root(doc) for doc in parents}
    groups = sorted({root(row.document_id) for row in eligible}, key=lambda value: sha256(f"{seed}\0{value}".encode()))
    if len(groups) < folds:
        raise ValueError("Not enough independent document groups for the requested folds")
    fold_for_group = {group: index % folds for index, group in enumerate(groups)}
    receipt = {"method": "parallel_document_normalized_duplicate_groups_v1", "seed": seed,
               "folds": folds, "max_chars_in_every_language": max_chars, "source_aligned_rows": len(records),
               "eligible_rows": len(eligible), "parallel_sentences": len(eligible) // len(LANGUAGES),
               "document_groups": len(groups), "exclusions": excluded,
               "group_for_document": group_for_document, "fold_for_group": fold_for_group,
               "labels_sha256": json_hash([asdict(row) for row in eligible]),
               "limitations": "Parallel translated text; custom grouped CV, not the official CoNLL test. No semantic near-duplicate detection beyond source-document/translation IDs and normalized exact text."}
    return GroupedCorpus(tuple(eligible), group_for_document, fold_for_group, receipt)
