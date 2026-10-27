"""GSTAR: extract reconstruction-query-block coverage from completed DEVELOPMENT runs.

No installation, model loading, training, new QA, masking, or held-out evaluation.
Uses the exact expanded-span-v2.1 prompt/code/checkpoints recorded by the experiment.

coverage_raw[l,h,c,i] = mean attention to ORIGINAL document key i, averaged over
query heads sharing KV head h and query rows of REPLAYED document block c.
The query token at each row has ALREADY been supplied (teacher forcing). This is
an attention diagnostic, NOT a reconstruction loss or a probability of correctness.

Blocks partition ONLY the replayed document query rows. Instruction/template rows
are EXCLUDED from coverage, but INCLUDED when reproducing the original g proxy.
All original document keys are saved. Protected positions are not removed here;
exclude them later when evaluating a free-slot selection constraint.

Target runtime: the user's pinned unquantized Qwen2.5 / Transformers 4.56.x.
API references:
https://huggingface.co/docs/transformers/v4.56.0/en/attention_interface
https://raw.githubusercontent.com/huggingface/transformers/v4.56.0/src/transformers/models/qwen2/modeling_qwen2.py
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

import logging

logger = logging.getLogger("gstar")

VERSION = "coverage-replayed-query-blocks-v1.0"
G_RTOL = 1e-4
G_ATOL = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


@contextmanager
def run_lock(path: Path):
    import fcntl  # Modal runs Linux; the OS releases this lock if the process ends.
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Coverage extraction is already running. Do not run two cells in parallel.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def block_ranges(length: int, n_blocks: int) -> list[tuple[int, int]]:
    require(type(n_blocks) is int and 1 <= n_blocks <= length,
            "n_blocks must be an integer between 1 and the document token count.")
    # Contiguous nonempty partition, each length differs by at most one token.
    return [(c * length // n_blocks, (c + 1) * length // n_blocks)
            for c in range(n_blocks)]


@torch.no_grad()
def reduce_layer_attention(weights: torch.Tensor, document_positions: list[int],
                           query_start: int, replay_start: int, total_tokens: int,
                           num_kv_heads: int, n_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure aggregation; returns CPU FP32 coverage[Hkv,C,D] and legacy g[Hkv,D]."""
    require(weights.ndim == 4 and weights.shape[0] == 1, "Expected batch=1 attention weights.")
    hq = weights.shape[1]
    require(hq % num_kv_heads == 0, "Invalid grouped-query attention configuration.")
    require(tuple(weights.shape[-2:]) == (total_tokens, total_tokens), "Attention shape mismatch.")
    require(0 <= max(document_positions) < query_start < replay_start < total_tokens,
            "Original-document / instruction / replay boundaries are inconsistent.")
    require(total_tokens - replay_start == len(document_positions), "Replay token count mismatch.")
    index = torch.tensor(document_positions, dtype=torch.long, device=weights.device)
    # The key axis refers to the ORIGINAL document, never its repeated copy.
    block = weights[0, :, query_start:, :].index_select(-1, index).float()
    require(bool(torch.isfinite(block).all() and (block >= 0).all()), "Invalid attention values.")
    grouped = block.reshape(num_kv_heads, hq // num_kv_heads,
                            total_tokens - query_start, len(document_positions))
    # Same definition as extract_g in the previously validated paired runner.
    g_recomputed = grouped.amax(dim=(1, 2)).cpu()
    replay_offset = replay_start - query_start
    coverage = torch.stack([
        grouped[:, :, replay_offset + start:replay_offset + end, :].mean(dim=(1, 2))
        for start, end in block_ranges(len(document_positions), n_blocks)
    ], dim=1).cpu()
    return coverage, g_recomputed


@torch.no_grad()
def collect_coverage(model: Any, input_ids: torch.Tensor, document_positions: list[int],
                     query_start: int, replay_start: int, n_blocks: int,
                     max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One full-causal forward, no cache object is created or reused."""
    require(model.config._attn_implementation == "eager", "Coverage requires eager attention.")
    require(input_ids.ndim == 2 and input_ids.shape[0] == 1, "Only batch size 1 is supported.")
    total = input_ids.shape[1]
    require(total <= max_tokens, f"Input has {total} tokens > {max_tokens}. No silent truncation.")
    ids = input_ids.to(model.device)
    output = None
    try:
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       output_attentions=True, return_dict=True, logits_to_keep=1)
        require(output.attentions is not None and
                len(output.attentions) == model.config.num_hidden_layers,
                "The model did not return the expected attention layers.")
        coverages, gs = [], []
        for weights in output.attentions:
            require(isinstance(weights, torch.Tensor), "An attention layer returned None.")
            require(weights.shape[1] == model.config.num_attention_heads, "Query head count mismatch.")
            c, g = reduce_layer_attention(weights, document_positions, query_start,
                                           replay_start, total, model.config.num_key_value_heads, n_blocks)
            coverages.append(c)
            gs.append(g)
        return torch.stack(coverages), torch.stack(gs)
    finally:
        del output  # Do not retain the large attention arrays in returned values.


def compare_g(new: torch.Tensor, old: torch.Tensor) -> dict:
    require(new.shape == old.shape, "Saved g and recomputed g shapes differ.")
    old = old.detach().float().cpu()
    new = new.detach().float().cpu()
    require(bool(torch.isfinite(old).all() and (old >= 0).all()), "Invalid saved g.")
    delta = (new - old).abs()
    return {"exact_match": torch.equal(new, old),
            "within_tolerance": torch.allclose(new, old, rtol=G_RTOL, atol=G_ATOL),
            "max_abs_difference": float(delta.max()), "mean_abs_difference": float(delta.mean()),
            "rtol": G_RTOL, "atol": G_ATOL}


def _hook_counts(model: Any) -> list[list[int]]:
    return [[len(x.self_attn._forward_pre_hooks), len(x.self_attn._forward_hooks)]
            for x in model.model.layers]


def _load_source_checkpoint(experiment: Path, doc_id: str) -> tuple[dict, Path, dict]:
    root = experiment / "runs" / "development" / doc_id
    marker = root / "done.json"
    require(marker.is_file(), f"No completed development checkpoint: {marker}")
    done = read_json(marker)
    require(done.get("status") == "COMPLETE", f"Incomplete development checkpoint: {doc_id}")
    attempt = (root / done["attempt"]).resolve()
    require(attempt.is_relative_to(root.resolve()), "Invalid checkpoint attempt path.")
    scores_path = attempt / "scores.pt"
    expected = done.get("file_hashes", {}).get("scores.pt")
    require(expected is not None and scores_path.is_file() and digest(scores_path) == expected,
            f"Missing or changed original scores: {scores_path}")
    scores = torch.load(scores_path, map_location="cpu", weights_only=True)
    return scores, scores_path, {"scores_sha256": expected, "done_sha256": digest(marker)}


def _input_groups(experiment: Path, protocol: dict) -> tuple[dict[str, list[dict]], str]:
    # Intentionally does NOT read evaluation_only, predictions, or any heldout file.
    path = experiment / "data" / "development" / "selection_inputs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        require(set(row) == {"case_id", "document_id", "document", "first_question"},
                "Unexpected selection-input fields.")
        groups[row["document_id"]].append(row)
    dev_ids = protocol["document_ids"]["development"]
    require(set(groups) == set(dev_ids), "Development document IDs have changed.")
    for doc_id, cases in groups.items():
        require(len(cases) == 2 and len({r["case_id"] for r in cases}) == 2,
                f"Expected two paired first-question inputs: {doc_id}")
        require(cases[0]["document"] == cases[1]["document"], "Paired documents differ.")
    return dict(groups), digest(path)


def _validate_layout_and_replay(model: Any, tokenizer: Any, paired: dict, render: Any,
                                rows: list[dict], scores: dict, max_tokens: int) -> tuple[dict, torch.Tensor, int]:
    document = rows[0]["document"]
    layout = paired["make_layout"](tokenizer, render, document)
    saved = scores["layout"]
    for key in ("document", "prefix_text", "prefix_length", "document_positions",
                "document_token_ids", "document_character_range"):
        require(layout[key] == saved[key], f"Saved/current layout mismatch: {key}")
    require(torch.equal(layout["prefix_ids"], saved["prefix_ids"]), "Saved prefix token IDs changed.")
    require(set(scores["cases"]) == {r["case_id"] for r in rows}, "Saved first-question cases changed.")
    shape = (model.config.num_hidden_layers, model.config.num_key_value_heads,
             len(layout["document_positions"]))
    require(tuple(scores["g_raw_proxy"].shape) == shape, "g shape/model mismatch.")
    for row in rows:
        a = scores["cases"][row["case_id"]]
        require(a["first_question"] == row["first_question"], "Saved first question changed.")
        require(tuple(a["a_raw"].shape) == shape and bool(torch.isfinite(a["a_raw"]).all())
                and bool((a["a_raw"] >= 0).all()), "Invalid saved a scores.")
    request = render(tokenizer, document, paired["RECONSTRUCTION_INSTRUCTION"])
    request_ids = paired["tokenized"](tokenizer, request)["input_ids"]
    p = layout["prefix_length"]
    require(torch.equal(request_ids[:, :p], layout["prefix_ids"]), "Reconstruction prefix changed.")
    replay = torch.tensor([layout["document_token_ids"]], dtype=torch.long)
    ids = torch.cat((request_ids, replay), dim=1)
    meta = scores["g_metadata"]
    require(meta["definition"] == "single_chunk_full_causal_reconstruction_max"
            and meta.get("native_kvzip_reproduction") is False, "Unexpected legacy g definition.")
    for key, value in {"query_start": p, "replay_start": request_ids.shape[1],
                       "total_tokens": ids.shape[1], "query_tokens": ids.shape[1] - p,
                       "instruction": paired["RECONSTRUCTION_INSTRUCTION"]}.items():
        require(meta[key] == value, f"Reconstruction metadata changed: {key}")
    require(ids.shape[1] <= max_tokens, f"Reconstruction exceeds {max_tokens} tokens; no truncation.")
    return layout, ids, request_ids.shape[1]


def extract_coverage(model: Any, tokenizer: Any, experiment_dir: str | Path,
                     document_ids: Sequence[str] | None = ("development_001",),
                     n_blocks: int = 4, max_new_documents: int | None = None) -> dict:
    """Extract only development coverage. None document_ids means all planned dev docs.

    Completed documents are hash-verified and skipped. Incomplete attempts are kept.
    Uses experiment/code snapshots, not mutable current-notebook globals or old prompts.
    Returns a summary with output_directory and per-document coverage file paths.
    """
    experiment = Path(experiment_dir).expanduser().resolve()
    protocol_path = experiment / "protocol.json"
    require(protocol_path.is_file(), f"Missing original protocol: {protocol_path}")
    protocol = read_json(protocol_path)
    require(protocol.get("version") == "expanded-span-v2.1", "Unexpected experiment version.")
    require(type(n_blocks) is int and 1 <= n_blocks <= 16, "Use an integer n_blocks in [1,16].")
    require(max_new_documents is None or
            (type(max_new_documents) is int and max_new_documents > 0), "Invalid max_new_documents.")
    expanded_path = experiment / "code" / "gstar_expanded_eval.py"
    paired_path = experiment / "code" / "paired.py"
    require(expanded_path.is_file() and digest(expanded_path) == protocol["script_sha256"],
            "Original expanded code snapshot is missing/changed.")
    require(paired_path.is_file() and digest(paired_path) == protocol["dependency_hashes"]["paired"],
            "Original paired code snapshot is missing/changed.")
    expanded = runpy.run_path(str(expanded_path))
    paired = runpy.run_path(str(paired_path))
    runtime = expanded["runtime_fingerprint"](model, tokenizer)
    changed = [k for k in set(runtime) | set(protocol["runtime"])
               if runtime.get(k) != protocol["runtime"].get(k)]
    require(not changed, f"Runtime differs from original: {changed}. Do not reinstall or bypass this check.")
    require(expanded["SYSTEM_PROMPT"] == protocol["system_prompt"], "Original system prompt changed.")
    require(hasattr(model, "set_attn_implementation"), "Attention backend switching is unavailable.")
    old_hooks = _hook_counts(model)
    require(all(x == [0, 0] for x in old_hooks),
            "Existing attention hooks detected. Stop other experiments before extracting coverage.")
    groups, input_sha = _input_groups(experiment, protocol)
    available = protocol["document_ids"]["development"]
    requested = list(available) if document_ids is None else list(document_ids)
    require(requested and len(requested) == len(set(requested)), "Select nonduplicate development document IDs.")
    require(all(x in available for x in requested), "Only original DEVELOPMENT IDs are allowed here.")
    requested = [x for x in available if x in requested]
    identity = {
        "version": VERSION, "extractor_sha256": digest(Path(__file__)),
        "source_experiment": str(experiment), "source_protocol_sha256": digest(protocol_path),
        "selection_inputs_sha256": input_sha, "source_runtime": runtime,
        "n_blocks": n_blocks, "query_scope": "replayed-document-token rows ONLY; instruction excluded",
        "key_scope": "all original document tokens, protected positions not yet excluded",
        "aggregation": "mean over grouped query heads and query tokens within each block",
        "block_rule": "boundaries floor(c * document_token_count / n_blocks)",
        "row_alignment": "attention after feeding each replayed document token (NOT next-token loss)",
        "g_recheck_scope": "instruction/template PLUS replay; legacy grouped-query maximum",
        "g_check_rtol": G_RTOL, "g_check_atol": G_ATOL,
        "split": "development", "heldout_read": False,
        "max_scoring_tokens": protocol["max_scoring_input_tokens"],
        "native_kvzip_reproduction": False, "normalized": False,
        "protected_positions_applied": False, "generates_answers": False,
    }
    out = experiment.parent / f"coverage_blocks_v1_b{n_blocks}_{object_digest(identity)[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out / ".run.lock"):
        meta_path = out / "coverage_protocol.json"
        if meta_path.is_file():
            require(read_json(meta_path) == identity, "Coverage configuration changed; do not overwrite.")
        else:
            save_json(meta_path, identity)
            shutil.copy2(__file__, out / "extractor_snapshot.py")
        old_backend, old_training = model.config._attn_implementation, model.training
        results, new_count, reused_count = [], 0, 0
        logger.debug('%s', "========== RECONSTRUCTION BLOCK COVERAGE ==========")
        logger.debug('%s', f"Development documents requested: {len(requested)} | blocks: {n_blocks}")
        logger.debug('%s %s', "Output:", out)
        logger.debug('%s', "Only scoring forward; no QA, no selector change, no held-out access.")
        try:
            model.eval()
            model.set_attn_implementation("eager")
            require(model.config._attn_implementation == "eager", "Could not switch to eager attention.")
            for doc_id in requested:
                scores, source_scores_path, fingerprint = _load_source_checkpoint(experiment, doc_id)
                doc_dir = out / doc_id
                done_path = doc_dir / "done.json"
                if done_path.is_file():
                    done = read_json(done_path)
                    require(done.get("status") == "COMPLETE" and done["source_fingerprint"] == fingerprint,
                            f"Original scores/checkpoint changed for {doc_id}.")
                    completed = (doc_dir / done["attempt"]).resolve()
                    require(completed.is_relative_to(doc_dir.resolve()), "Invalid output checkpoint path.")
                    for name, expected in done["file_hashes"].items():
                        require(digest(completed / name) == expected, f"Coverage output changed: {completed/name}")
                    result = read_json(completed / "summary.json")
                    results.append(result); reused_count += 1
                    logger.debug('%s', f"  {doc_id}: verified existing coverage; skipped forward.")
                    continue
                if max_new_documents is not None and new_count >= max_new_documents:
                    continue
                attempt = doc_dir / f"attempt_{stamp()}"
                attempt.mkdir(parents=True, exist_ok=False)
                try:
                    layout, ids, replay_start = _validate_layout_and_replay(
                        model, tokenizer, paired, expanded["render_prompt"], groups[doc_id], scores,
                        protocol["max_scoring_input_tokens"])
                    d = len(layout["document_positions"])
                    ranges = block_ranges(d, n_blocks)
                    logger.debug('%s', f"  {doc_id}: document={d}, total={ids.shape[1]}, "
                          f"block sizes={[e-s for s,e in ranges]} ...")
                    coverage, g_new = collect_coverage(
                        model, ids, layout["document_positions"], layout["prefix_length"],
                        replay_start, n_blocks, protocol["max_scoring_input_tokens"])
                    expected_shape = (model.config.num_hidden_layers, model.config.num_key_value_heads, n_blocks, d)
                    require(tuple(coverage.shape) == expected_shape, "Coverage shape mismatch.")
                    require(bool(torch.isfinite(coverage).all() and (coverage >= 0).all()), "Invalid coverage scores.")
                    mass = coverage.double().sum(-1)
                    require(float(mass.max()) <= 1.02, "Coverage row sum exceeds attention-mass range.")
                    check = compare_g(g_new, scores["g_raw_proxy"])
                    save_json(attempt / "g_recheck.json", check)
                    require(check["within_tolerance"],
                            f"Recomputed g does not match source g: {check}. Inspect {attempt/'g_recheck.json'}.")
                    target = attempt / "coverage.pt"
                    payload = {
                        "version": VERSION, "document_id": doc_id, "split": "development",
                        "coverage_raw": coverage, "g_recomputed_for_check": g_new,
                        "document_positions": list(layout["document_positions"]),
                        "document_token_ids": list(layout["document_token_ids"]),
                        "prefix_input_ids": layout["prefix_ids"], "reconstruction_input_ids": ids.cpu(),
                        "prefix_length": layout["prefix_length"], "replay_start": replay_start,
                        "replay_relative_block_ranges": ranges,
                        "absolute_query_block_ranges": [(replay_start+s, replay_start+e) for s,e in ranges],
                        "n_blocks": n_blocks, "coverage_definition": identity["aggregation"],
                        "query_scope": identity["query_scope"], "row_alignment": identity["row_alignment"],
                        "normalized": False, "protected_positions_applied": False,
                        "source_scores_path": str(source_scores_path), "source_fingerprint": fingerprint,
                        "source_protocol_sha256": identity["source_protocol_sha256"],
                        "source_runtime": runtime, "g_recheck": check,
                    }
                    torch.save(payload, target)
                    # Check the actual bytes can be read safely before marking completion.
                    saved = torch.load(target, map_location="cpu", weights_only=True)
                    require(torch.equal(saved["coverage_raw"], coverage), "Coverage save/load verification failed.")
                    result = {
                        "document_id": doc_id, "status": "COMPLETE", "shape": list(coverage.shape),
                        "document_tokens": d, "total_input_tokens": ids.shape[1],
                        "replay_query_tokens": d, "instruction_template_query_tokens": replay_start-layout["prefix_length"],
                        "block_token_counts": [e-s for s,e in ranges],
                        "finite": True, "nonnegative": True,
                        "block_document_mass_min": float(mass.min()),
                        "block_document_mass_max": float(mass.max()),
                        "zero_mass_layer_head_blocks": int((mass == 0).sum()),
                        "g_recheck": check, "coverage_file": str(target),
                        "original_scores_unmodified": digest(source_scores_path) == fingerprint["scores_sha256"],
                    }
                    require(result["original_scores_unmodified"], "Original score file changed during extraction.")
                    save_json(attempt / "summary.json", result)
                    save_json(done_path, {
                        "status": "COMPLETE", "attempt": attempt.name, "source_fingerprint": fingerprint,
                        "file_hashes": {name: digest(attempt/name)
                                        for name in ("coverage.pt", "summary.json", "g_recheck.json")},
                    })
                    results.append(result); new_count += 1
                    logger.debug('%s', f"    shape={tuple(coverage.shape)} | g exact={check['exact_match']} | "
                          f"g within tolerance={check['within_tolerance']} | saved")
                except BaseException as exc:
                    save_json(attempt / "failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
                    logger.error('%s %s', "Stopped without modifying original experiment. Error:", attempt/"failure.json")
                    raise
        finally:
            try:
                model.set_attn_implementation(old_backend)
            finally:
                model.train(old_training)
                restored = {
                    "attention_backend_restored": model.config._attn_implementation == old_backend,
                    "backend_after": model.config._attn_implementation,
                    "training_mode_restored": model.training == old_training,
                    "hook_counts_unchanged": _hook_counts(model) == old_hooks,
                    "source_protocol_unchanged": digest(protocol_path) == identity["source_protocol_sha256"],
                }
                save_json(out / f"restoration_{stamp()}.json", restored)
        require(all(restored[k] for k in ("attention_backend_restored", "training_mode_restored",
                                         "hook_counts_unchanged", "source_protocol_unchanged")),
                "Model/source settings were not fully restored; inspect restoration file.")
        summary = {
            "status": "COMPLETE" if len(results) == len(requested) else "INCOMPLETE",
            "output_directory": str(out), "source_experiment": str(experiment),
            "requested_documents": requested, "completed_requested": len(results),
            "newly_extracted": new_count, "verified_reused": reused_count, "n_blocks": n_blocks,
            "all_g_checks_pass": all(r["g_recheck"]["within_tolerance"] for r in results),
            "documents": results, **restored,
            "scope": "Coverage extraction only; no accuracy claim, no heldout evaluation.",
        }
        save_json(out / f"run_summary_{stamp()}.json", summary)
        logger.debug('%s', "\n========== COVERAGE EXTRACTION SUMMARY ==========")
        logger.debug('%s', f"Status: {summary['status']} | completed: {len(results)}/{len(requested)}")
        logger.debug('%s', f"Newly extracted: {new_count} | verified reuse: {reused_count}")
        logger.debug('%s %s', "All original g checks within tolerance:", summary["all_g_checks_pass"])
        logger.debug('%s %s', "Attention backend restored:", restored["backend_after"])
        logger.debug('%s %s', "Original protocol unchanged:", restored["source_protocol_unchanged"])
        logger.info('%s %s', "Saved under:", out)
        logger.debug('%s', "Coverage only. Next step is a separate selector implementation/evaluation.")
        return summary
