"""Bounded, pinned native Ukrainian and translated Polish/Czech UD input.

This profile deliberately leaves the historical EN/JA/ZH PUD contract alone.
Source bytes are never normalized or repaired. Native Ukrainian dates are split
before duplicate removal so recurrent parliamentary phrases cannot join every
session into one group. No network access occurs on import or without opt-in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from urllib.request import urlopen

from .external_corpus import json_hash, normalized_text, sha256


SLAVIC_LANGUAGES = ("uk", "pl", "cs")
MAX_SOURCE_BYTES = 32 * 1024 * 1024
_NATIVE_SOURCE = "UD_Ukrainian-ParlaMint"
_SOURCES = {"uk": _NATIVE_SOURCE, "pl": "UD_Polish-PUD", "cs": "UD_Czech-PUD"}
_UK_SENTENCE = re.compile(r"^(ParlaMint-UA_\d{4}-\d{2}-\d{2})-m\d+\.u\d+(?:\.|$)")
_NSDC_SENTENCE = re.compile(r"^(NSDC_UA_28_Feb2014)-[1-9]\d*$")


def _native_document_id(sentence_id):
    # The pinned README explicitly includes this single NSDC meeting alongside
    # the Rada plenary dates. This is a source transcript ID, not a fallback.
    match = _UK_SENTENCE.match(sentence_id) or _NSDC_SENTENCE.fullmatch(sentence_id)
    if not match:
        raise ValueError("Ukrainian sentence ID must contain a documented native date/session")
    return match.group(1)


@dataclass(frozen=True)
class SlavicSentence:
    language: str
    document_id: str
    sentence_id: str
    text: str
    segments: tuple[str, ...]
    upstream_split: str
    source_id: str
    parallel_id: str | None = None


def _surface_forms(rows):
    forms, covered_until, previous_id = [], 0, 0
    for fields in rows:
        token_id, form = fields[:2]
        if re.fullmatch(r"(?:0|[1-9]\d*)\.[1-9]\d*", token_id):
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
    return forms


def _align(text, forms):
    if not forms:
        return None
    segments, cursor = [], 0
    for form in forms:
        start = text.find(form, cursor) if form else -1
        gap = text[cursor:start] if start >= 0 else ""
        if start < 0 or (gap and not gap.isspace()):
            return None
        if gap:
            segments.append(gap)
        segments.append(form)
        cursor = start + len(form)
    suffix = text[cursor:]
    if suffix and not suffix.isspace():
        return None
    if suffix:
        segments.append(suffix)
    return tuple(segments) if "".join(segments) == text else None


def parse_slavic_conllu(content, language, *, source_id, upstream_split,
                        pud_document_for_parallel=None):
    """Read exact surface spans and explicit provenance; report unalignable rows.

    ParlaMint sentence IDs encode actual source dates even when an upstream dev
    file begins inside a document. PUD requires the caller's original EN/JA/ZH
    parallel-ID-to-document map. It never infers documents from sentence IDs.
    """
    if language not in SLAVIC_LANGUAGES or source_id != _SOURCES[language]:
        raise ValueError("Unsupported language/source identity")
    if upstream_split not in ("train", "dev", "test"):
        raise ValueError("Unknown upstream split")
    if language != "uk" and pud_document_for_parallel is None:
        raise ValueError("PUD input requires original parallel document metadata")
    records, excluded, seen = [], [], set()
    explicit_document = None
    # Splitting only LF keeps Unicode line separators inside the actual text.
    for block in re.split(r"\n[ \t]*\r?\n", content):
        metadata, rows = {}, []
        for raw_line in block.split("\n"):
            line = raw_line.removesuffix("\r")
            if line.startswith("#"):
                key, separator, value = line[1:].lstrip().partition(" = ")
                if separator:
                    key = key.strip()
                    if key in metadata:
                        raise ValueError("Duplicate CoNLL-U metadata key")
                    metadata[key] = value
            elif line.strip():
                fields = line.split("\t")
                if len(fields) != 10:
                    raise ValueError("CoNLL-U token rows must have ten fields")
                rows.append(fields)
        newdoc = metadata.get("newdoc id", metadata.get("newdoc_id"))
        if "newdoc id" in metadata and "newdoc_id" in metadata and metadata["newdoc id"] != metadata["newdoc_id"]:
            raise ValueError("Conflicting document metadata aliases")
        if newdoc is not None:
            explicit_document = newdoc
        if not rows:
            continue
        sentence_id, text = metadata.get("sent_id"), metadata.get("text")
        if not sentence_id or text is None or not text:
            raise ValueError("Explicit sentence and nonempty text metadata are required")
        if sentence_id in seen:
            raise ValueError("Duplicate sentence ID")
        seen.add(sentence_id)
        if language == "uk":
            document_id = _native_document_id(sentence_id)
            if newdoc and not (newdoc == document_id or newdoc.startswith(document_id + "-m")):
                raise ValueError("ParlaMint document metadata disagrees with sentence date")
            parallel_id = None
            if metadata.get("parallel_id"):
                raise ValueError("Native Ukrainian input must not invent translation IDs")
        else:
            parallel_id = metadata.get("parallel_id")
            if not parallel_id or parallel_id not in pud_document_for_parallel:
                raise ValueError("PUD parallel ID lacks original document provenance")
            document_id = pud_document_for_parallel[parallel_id]
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("PUD original document provenance must be a nonempty ID")
            if explicit_document and explicit_document != document_id:
                raise ValueError("PUD document metadata disagrees with original document provenance")
        segments = _align(text, _surface_forms(rows))
        if segments is None:
            excluded.append({"language": language, "source_id": source_id,
                             "upstream_split": upstream_split, "document_id": document_id,
                             "sentence_id": sentence_id, "reason": "surface_form_alignment",
                             "text_sha256": sha256(text.encode("utf-8"))})
            continue
        records.append(SlavicSentence(language, document_id, sentence_id, text,
                                      segments, upstream_split, source_id, parallel_id))
    return records, excluded


def _validated_manifest(manifest):
    sources = manifest.get("sources", [])
    if sorted(source.get("language", "") for source in sources) != sorted(SLAVIC_LANGUAGES):
        raise ValueError("Slavic manifest must contain exactly Ukrainian, Polish, and Czech")
    for source in sources:
        language, revision = source["language"], source.get("revision", "")
        source_id = source.get("source_id")
        if source_id != _SOURCES[language] or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Source identity and immutable Git revision are required")
        if source.get("license") != "CC-BY-SA-4.0":
            raise ValueError("This bounded profile requires the pinned CC-BY-SA-4.0 sources")
        files = source.get("files", {})
        expected_data = ({f"uk_parlamint-ud-{role}.conllu" for role in ("train", "dev", "test")}
                         if language == "uk" else {f"{language}_pud-ud-test.conllu"})
        if set(files) != expected_data | {"README.md", "LICENSE.txt"}:
            raise ValueError("Manifest must pin all expected data, attribution, and license files")
        for filename, item in files.items():
            if Path(filename).name != filename or "\\" in filename:
                raise ValueError("Source filenames must be basenames")
            expected_url = f"https://raw.githubusercontent.com/UniversalDependencies/{source_id}/{revision}/{filename}"
            if item.get("url") != expected_url:
                raise ValueError("Download URL must match the exact pinned upstream source")
            size = item.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_SOURCE_BYTES:
                raise ValueError("Source byte length exceeds the acquisition bound")
            if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
                raise ValueError("Every source file requires a SHA256 digest")
            if filename.endswith(".conllu"):
                role = filename.rsplit("-", 1)[1].split(".")[0]
                if item.get("upstream_split") != role:
                    raise ValueError("Data file and explicit upstream split disagree")
    return sources


def load_slavic_sources(manifest_path, cache_dir, *, pud_document_for_parallel, download=False):
    """Acquire at most one request per missing file, then verify every byte/hash."""
    manifest_bytes = Path(manifest_path).read_bytes()
    sources = _validated_manifest(json.loads(manifest_bytes))
    records, exclusions, receipts = [], [], []
    seen = set()
    for source in sources:
        language, revision = source["language"], source["revision"]
        source_records, source_excluded, file_receipts = [], [], []
        for filename, item in source["files"].items():
            destination = Path(cache_dir) / language / revision / filename
            acquired = False
            if not destination.exists():
                if not download:
                    raise FileNotFoundError(f"Missing source {filename}; use explicit download=True")
                with urlopen(item["url"], timeout=45) as response:
                    payload = response.read(item["bytes"] + 1)
                if len(payload) != item["bytes"] or sha256(payload) != item["sha256"]:
                    raise ValueError(f"Source integrity mismatch: {filename}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                acquired = True
            if destination.stat().st_size != item["bytes"]:
                raise ValueError(f"Source integrity mismatch: {filename}")
            payload = destination.read_bytes()
            if len(payload) != item["bytes"] or sha256(payload) != item["sha256"]:
                raise ValueError(f"Source integrity mismatch: {filename}")
            file_receipts.append({"filename": filename, **item, "downloaded": acquired})
            if filename.endswith(".conllu"):
                parsed, rejected = parse_slavic_conllu(
                    payload.decode("utf-8"), language, source_id=source["source_id"],
                    upstream_split=item["upstream_split"],
                    pud_document_for_parallel=pud_document_for_parallel)
                for row in [*parsed, *rejected]:
                    key = ((row["language"], row["sentence_id"]) if isinstance(row, dict)
                           else (row.language, row.sentence_id))
                    if key in seen:
                        raise ValueError("Sentence ID repeats across upstream splits")
                    seen.add(key)
                source_records.extend(parsed)
                source_excluded.extend(rejected)
        records.extend(source_records)
        exclusions.extend(source_excluded)
        receipts.append({**{key: value for key, value in source.items() if key != "files"},
                         "files": file_receipts, "aligned_sentences": len(source_records),
                         "unalignable_sentences": len(source_excluded),
                         "document_groups": len({row.document_id for row in source_records}),
                         "per_upstream_split": {role: sum(row.upstream_split == role for row in source_records)
                                                for role in ("train", "dev", "test")}})
    return records, {"source_manifest_sha256": sha256(manifest_bytes), "sources": receipts,
                     "alignment_exclusions": exclusions,
                     "tokenization": "Exact UD surface MWT spans; empty nodes ignored; Unicode whitespace retained",
                     "labels_sha256": json_hash([asdict(row) for row in records])}


def partition_ukrainian(records, *, seed=5042, folds=5, test_fold=0, max_chars=None):
    """Pool official roles, split real dates, and remove duplicates across roles.

    This accepts only native Ukrainian rows. Apply length filtering and study
    caps only after source-date assignment and global duplicate exclusion.
    """
    records = tuple(records)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 3:
        raise ValueError("At least three integer folds are required")
    if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 2):
        raise ValueError("max_chars must be an integer of at least two")
    seen, text_labels = set(), {}
    for row in records:
        if row.language != "uk" or row.source_id != _NATIVE_SOURCE or row.parallel_id is not None:
            raise ValueError("Native partition accepts only original Ukrainian source rows")
        if row.document_id != _native_document_id(row.sentence_id):
            raise ValueError("Native partition requires original ParlaMint date provenance")
        if row.sentence_id in seen:
            raise ValueError("Duplicate sentence ID")
        seen.add(row.sentence_id)
        if not row.segments or any(not part for part in row.segments) or "".join(row.segments) != row.text:
            raise ValueError("Labels must reconstruct the source text exactly")
        if row.text in text_labels and text_labels[row.text] != row.segments:
            raise ValueError("Identical source text has conflicting supervision")
        text_labels[row.text] = row.segments
    documents = sorted({row.document_id for row in records},
                       key=lambda doc: sha256(f"{seed}\0{doc}".encode("utf-8")))
    if len(documents) < 3:
        raise ValueError("At least three independent ParlaMint dates are required")
    effective_folds = min(folds, len(documents))
    if isinstance(test_fold, bool) or not isinstance(test_fold, int) or not 0 <= test_fold < effective_folds:
        raise ValueError("test fold is outside the frozen native partition")
    dev_fold = (test_fold + 1) % effective_folds
    fold_for_document = {doc: index % effective_folds for index, doc in enumerate(documents)}
    role_for_document = {doc: "test" if fold == test_fold else "dev" if fold == dev_fold else "train"
                         for doc, fold in fold_for_document.items()}
    duplicate_rows = {}
    for row in records:
        duplicate_rows.setdefault(normalized_text(row.text), []).append(row)
    partitions = {name: [] for name in ("train", "dev", "test")}
    exclusions = []
    for key, duplicates in sorted(duplicate_rows.items()):
        duplicates.sort(key=lambda row: (row.document_id, row.sentence_id, row.upstream_split))
        roles = {role_for_document[row.document_id] for row in duplicates}
        cross_role = len(roles) > 1
        for index, row in enumerate(duplicates):
            role = role_for_document[row.document_id]
            reason = ("normalized_duplicate_cross_role" if cross_role else
                      "normalized_duplicate_within_role" if index else
                      "length_filter" if max_chars is not None and not 2 <= len(row.text) <= max_chars else None)
            if reason:
                exclusions.append({"source_id": row.source_id, "sentence_id": row.sentence_id,
                                   "document_id": row.document_id, "upstream_split": row.upstream_split,
                                   "role": role, "reason": reason,
                                   "normalized_text_sha256": sha256(key.encode("utf-8"))})
            else:
                partitions[role].append(row)
    for role in partitions:
        partitions[role] = tuple(sorted(partitions[role], key=lambda row: (row.document_id, row.sentence_id)))
    if any(not rows for rows in partitions.values()):
        raise ValueError("All native train/dev/test partitions must remain nonempty after exclusions")
    receipt = {"method": "native_ParlaMint_date_split_then_duplicate_exclusion_v1",
               "seed": seed, "requested_folds": folds, "folds": effective_folds,
               "test_fold": test_fold, "dev_fold": dev_fold, "max_chars": max_chars,
               "source_rows": len(records), "document_groups": len(documents),
               "fold_for_document": fold_for_document, "role_for_document": role_for_document,
               "exclusions": exclusions,
               "limitations": "Native parliamentary speech boundary learning; upstream roles pooled into custom date-grouped folds. Normalized exact duplicate exclusion does not detect semantic near duplicates."}
    for role, rows in partitions.items():
        receipt[role] = {"samples": len(rows), "documents": sorted({row.document_id for row in rows}),
                         "sentence_ids": [row.sentence_id for row in rows],
                         "labels_sha256": json_hash([asdict(row) for row in rows])}
    return partitions, receipt
