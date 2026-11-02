"""Pinned evaluation-only CoNLL-U corpora, with honest document provenance."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from urllib.request import urlopen

from .external_corpus import normalized_text, sha256
from .slavic_corpus import _align, _surface_forms


@dataclass(frozen=True)
class HeldoutSentence:
    language: str
    source_id: str
    sentence_id: str
    document_id: str | None
    text: str
    segments: tuple[str, ...]


def parse_heldout_conllu(content, language, source_id):
    """Retain source documents only when explicitly supplied, never inferred.

    A missing document does not remove an otherwise aligned sentence; callers
    must label that uncertainty and omit document-bootstrap intervals for it.
    Unicode, whitespace and UD surface multiword spans remain unchanged.
    """
    records, rejected, seen = [], [], set()
    document = None
    for block in re.split(r"\n[ \t]*\r?\n", content):
        metadata, rows = {}, []
        for raw in block.split("\n"):
            line = raw.removesuffix("\r")
            if line.startswith("#"):
                key, separator, value = line[1:].lstrip().partition(" = ")
                if separator:
                    metadata[key.strip().replace(" ", "_")] = value
                elif line.strip() == "# newdoc":
                    document = None
            elif line.strip():
                fields = line.split("\t")
                if len(fields) != 10:
                    raise ValueError("CoNLL-U rows require ten fields")
                rows.append(fields)
        if "newdoc_id" in metadata:
            document = metadata["newdoc_id"] or None
        if not rows:
            continue
        sentence = metadata.get("sent_id")
        text = metadata.get("text")
        if not sentence or text is None or sentence in seen:
            raise ValueError("Unique sentence IDs and original text are required")
        seen.add(sentence)
        segments = _align(text, _surface_forms(rows))
        if segments is None:
            rejected.append({"language": language, "source_id": source_id,
                             "sentence_id": sentence, "reason": "unalignable_surface_tokens"})
        else:
            records.append(HeldoutSentence(language, source_id, sentence, document, text, segments))
    return records, rejected


def load_heldout_sources(manifest_path, cache_dir, *, download=False):
    """Read immutable test files and their licenses; no implicit network retry."""
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported held-out manifest")
    records, sources, rejected = [], [], []
    languages = set()
    for source in manifest["sources"]:
        language, revision, repo = source["language"], source["revision"], source["source_id"]
        if language in languages or not re.fullmatch(r"[a-z]{2}", language) or not re.fullmatch(r"UD_[A-Za-z-]+", repo):
            raise ValueError("Invalid or repeated source identity")
        languages.add(language)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Source requires immutable revision")
        files = source["files"]
        data_files = [name for name in files if name.endswith("-ud-test.conllu")]
        if len(data_files) != 1 or set(files) != {"README.md", "LICENSE.txt", *data_files}:
            raise ValueError("Pin one official test file, README and LICENSE")
        contents = {}
        for filename, info in files.items():
            expected = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{revision}/{filename}"
            if (Path(filename).name != filename or "\\" in filename or info["url"] != expected
                    or isinstance(info["bytes"], bool) or not isinstance(info["bytes"], int)
                    or not 0 < info["bytes"] <= 32 * 1024 * 1024
                    or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])):
                raise ValueError("Invalid pinned file identity or acquisition bound")
            path = Path(cache_dir) / language / revision / filename
            if not path.exists():
                if not download:
                    raise FileNotFoundError("Missing held-out source; explicitly enable download: " + filename)
                with urlopen(expected, timeout=45) as response:
                    payload = response.read(info["bytes"] + 1)
                if len(payload) != info["bytes"] or sha256(payload) != info["sha256"]:
                    raise ValueError("Held-out download integrity mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            payload = path.read_bytes()
            if len(payload) != info["bytes"] or sha256(payload) != info["sha256"]:
                raise ValueError("Held-out cache integrity mismatch")
            contents[filename] = payload.decode("utf-8")
        parsed, excluded = parse_heldout_conllu(contents[data_files[0]], language, repo)
        records.extend(parsed)
        rejected.extend(excluded)
        sources.append({**source, "aligned": len(parsed), "missing_document_ids": sum(row.document_id is None for row in parsed),
                        "source_documents": len({row.document_id for row in parsed if row.document_id is not None})})
    return records, {"manifest_sha256": sha256(manifest_bytes), "sources": sources, "alignment_exclusions": rejected}


def select_unseen(records, prior_texts, *, cap, max_chars, seed):
    """Exclude ALL old-source overlap and duplicates before length/case caps."""
    prior = {normalized_text(text) for text in prior_texts}
    buckets, rejected = {}, []
    for row in records:
        buckets.setdefault(normalized_text(row.text), []).append(row)
    eligible = []
    for key, rows in sorted(buckets.items()):
        rows.sort(key=lambda row: (row.language, row.source_id, row.sentence_id))
        ambiguous = len({row.segments for row in rows}) > 1
        for index, row in enumerate(rows):
            reason = ("overlap_with_any_prior_source" if key in prior else
                      "ambiguous_normalized_supervision" if ambiguous else
                      "normalized_duplicate" if index else
                      "length_filter" if not 2 <= len(row.text) <= max_chars else None)
            if reason:
                rejected.append({"language": row.language, "source_id": row.source_id, "sentence_id": row.sentence_id,
                                 "reason": reason, "normalized_text_sha256": sha256(key.encode())})
            else:
                eligible.append(row)
    selected = []
    for language in sorted({row.language for row in records}):
        choices = [row for row in eligible if row.language == language]
        choices.sort(key=lambda row: sha256(f"{seed}\0{row.source_id}\0{row.sentence_id}".encode()))
        if len(choices) < cap:
            raise ValueError("Insufficient unseen cases for " + language)
        selected.extend(choices[:cap])
    return selected, {"exclusions": rejected, "available_per_language": {
        language: sum(row.language == language for row in eligible) for language in sorted({row.language for row in records})}}
