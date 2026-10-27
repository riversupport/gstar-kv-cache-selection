"""Frozen, article-disjoint FRESH TEST for GSTAR coverage eta=.95.

Uses the validated source snapshots; does NOT edit old experiments or tune on test.
API: prepare_fresh_test -> run_fresh_test (resumable) -> export_review.
Four distinct intended policies, five report labels:
  full_kv, g_proxy_only, fixed_tuned (V1 development winner),
  fixed_lambda_0.75 (prespecified trade-off comparator), coverage_eta_0.95.
The original dev-tuned fixed endpoint is kept even when it aliases g-only.

No training, package installation, model download, or native KVzip/speed claim.
Preparation requires the original cached SQuAD 1.1 JSON. Source checkpoints and
code snapshots are checked. New article selection uses only prior membership,
original article order, original builder, and the original length/structure rules.
"""
from __future__ import annotations

import copy
import csv
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
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

import logging

logger = logging.getLogger("gstar")

VERSION = "fresh-coverage-test-v1.0"
ETA = 0.95
COMPARISON_LAMBDA = 0.75
BLOCKS = 4
BOOTSTRAP_SEED = 20261023
BOOTSTRAP_REPETITIONS = 10000
METHODS = ("full_kv", "g_proxy_only", "fixed_tuned", "fixed_lambda_0.75", "coverage_eta_0.95")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def objhash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


@contextmanager
def run_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another cell is using this experiment. Do not run concurrently.") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def resolve_recorded_dir(recorded: str, siblings: Path, marker: str) -> Path:
    direct = Path(recorded)
    for path in (direct, siblings / direct.name):
        if (path / marker).is_file():
            return path.resolve()
    raise FileNotFoundError(f"Recorded directory missing: {recorded} ({marker})")


def load_original_tools(source: Path, protocol: dict) -> tuple[dict, dict[str, Path]]:
    paths = {"expanded": source/"code/gstar_expanded_eval.py"}
    paths.update({k: source/"code"/(k + ".py") for k in ("paired", "smoke", "full", "builder")})
    expected = {"expanded": protocol["script_sha256"], **protocol["dependency_hashes"]}
    for key, path in paths.items():
        require(path.is_file() and digest(path) == expected[key], f"Original code changed/missing: {path}")
    return {k: runpy.run_path(str(p)) for k, p in paths.items()}, paths


def discover_selector(source: Path, supplied: str | Path | None) -> Path:
    candidates = [Path(supplied)] if supplied is not None else list(source.parent.glob("coverage_eval_v1_b4_*"))
    matched = []
    source_hash = digest(source/"protocol.json")
    for root in candidates:
        path = root/"selector_protocol.json"
        if not path.is_file():
            continue
        meta = read_json(path)
        if meta.get("source_protocol_sha256") != source_hash:
            continue
        summary_file = root/"evaluation/summary.json"
        if not summary_file.is_file():
            continue
        summary = read_json(summary_file)
        if summary.get("status") == "COMPLETE":
            matched.append(root.resolve())
    require(len(matched) == 1,
            f"Need exactly one completed coverage development run; found {len(matched)}. "
            "Pass coverage_selector_dir explicitly instead of choosing a different source silently.")
    return matched[0]


def known_articles(project: Path, exclude_root: Path | None = None,
                   extra_dirs: tuple[Path, ...] = ()) -> dict:
    """Read MEMBERSHIP/INPUTS only, never model predictions, to exclude prior articles.

    Includes prepared/reserved sets even when not evaluated yet. Outside-project
    work cannot be discovered: pass its data directories via extra exclusion dirs.
    """
    paths: set[Path] = set()
    patterns = (
        "experiments/*/data/*/evaluation_only.jsonl",
        "experiments/*/source_development/evaluation_only.jsonl",
        "results/*/data_snapshot/evaluation_only.jsonl",
        "data/pilot/evaluation_only.jsonl",
        "gstar_fullkv_pilot/data/pilot/evaluation_only.jsonl",
    )
    for pattern in patterns:
        paths.update(p.resolve() for p in project.glob(pattern))
    for root in extra_dirs:
        require(root.is_dir(), f"Extra exclusion directory missing: {root}")
        paths.update(p.resolve() for p in root.rglob("evaluation_only.jsonl"))
    titles, qids, documents = set(), set(), set()
    files = []
    for path in sorted(paths):
        if exclude_root is not None and path.is_relative_to(exclude_root.resolve()):
            continue
        labs = read_jsonl(path)
        for row in labs:
            title = row.get("source_article")
            require(isinstance(title, str) and bool(title), f"Missing article identity in {path}")
            titles.add(title)
            for key in ("first_question_reference", "target_question_reference"):
                ref = row.get(key)
                require(isinstance(ref, dict) and ref.get("question_id"), f"Missing QA identity in {path}")
                qids.add(ref["question_id"])
        entry = {"path": str(path), "sha256": digest(path)}
        inp = path.with_name("selection_inputs.jsonl")
        if inp.is_file():
            for row in read_jsonl(inp):
                require(isinstance(row.get("document"), str), f"Missing document in {inp}")
                documents.add(hashlib.sha256(row["document"].encode()).hexdigest())
            entry["inputs_sha256"] = digest(inp)
        files.append(entry)
    return {"article_titles": sorted(titles), "question_ids": sorted(qids),
            "document_sha256": sorted(documents), "membership_files": files}


def assert_disjoint(packs: list[dict], exclusions: dict) -> None:
    old_titles = set(exclusions["article_titles"])
    old_ids = set(exclusions["question_ids"])
    old_docs = set(exclusions["document_sha256"])
    titles = [p["source_article"] for p in packs]
    require(len(set(titles)) == len(titles), "Fresh test must have one pack per source article.")
    require(not set(titles) & old_titles, "A previously used/reserved article entered the fresh test.")
    for p in packs:
        require(hashlib.sha256(p["document"].encode()).hexdigest() not in old_docs,
                "A document duplicates a previously used input.")
        for key in ("target", "same_first", "different_first"):
            require(p[key]["question_id"] not in old_ids, "A previously used question entered the fresh test.")


def _source_bundle(source: Path, selector: Path, model: Any, tokenizer: Any) -> tuple[dict, dict, dict]:
    protocol = read_json(source/"protocol.json")
    require(protocol["version"] == "expanded-span-v2.1", "Unexpected original experiment version.")
    tools, paths = load_original_tools(source, protocol)
    tools["expanded"]["_validate_data"](source, protocol)
    require(tools["expanded"]["runtime_fingerprint"](model, tokenizer) == protocol["runtime"],
            "Model/tokenizer/runtime differs from the original. Do not reinstall or bypass the check.")
    require(protocol["keep_fraction"] == .5 and protocol["max_new_tokens"] == 64,
            "This follow-up inherits the original 50% budget and 64-token cap.")
    meta = read_json(selector/"selector_protocol.json")
    require(meta["source_protocol_sha256"] == digest(source/"protocol.json"), "Wrong selector source.")
    require(meta["n_blocks"] == BLOCKS and not meta["total_g_constraint_also_applied"], "Coverage definition changed.")
    covdir = resolve_recorded_dir(meta["coverage_directory"], source.parent, "coverage_protocol.json")
    cp = read_json(covdir/"coverage_protocol.json")
    require(digest(covdir/"coverage_protocol.json") == meta["coverage_protocol_sha256"], "Coverage protocol changed.")
    require(cp["source_protocol_sha256"] == digest(source/"protocol.json") and cp["n_blocks"] == BLOCKS,
            "Wrong coverage extraction source.")
    paths["coverage"] = selector/"runner_snapshot.py"
    paths["extractor"] = covdir/"extractor_snapshot.py"
    require(digest(paths["coverage"]) == meta["runner_sha256"], "Coverage selector snapshot changed.")
    require(digest(paths["extractor"]) == cp["extractor_sha256"], "Coverage extractor snapshot changed.")
    tools.update({k: runpy.run_path(str(paths[k])) for k in ("coverage", "extractor")})
    csum = read_json(selector/"evaluation/summary.json")
    require(csum["status"] == "COMPLETE" and csum["documents_completed"] == protocol["n_development"],
            "Finish the prespecified coverage development set first.")
    require(csum.get("coverage_selected_on_development") == "coverage_eta_0.95",
            "The recorded coverage development choice is not eta=.95; inspect instead of silently changing it.")
    original_frozen = read_json(source/"frozen_protocol.json")
    require(original_frozen["protocol_sha256"] == digest(source/"protocol.json"), "Original freeze mismatch.")
    require(original_frozen["development_summary_sha256"] == digest(source/"runs/development/summary.json"),
            "Original development summary changed after freeze.")
    fixed = original_frozen["selected"]["fixed"]
    allowed = {"a_only", "fixed_lambda_0.25", "fixed_lambda_0.50", "fixed_lambda_0.75", "g_proxy_only"}
    require(fixed in allowed, "Unknown original development-tuned fixed baseline.")
    provenance = {
        "source_experiment": str(source), "source_protocol_sha256": digest(source/"protocol.json"),
        "coverage_selector_dir": str(selector), "selector_protocol_sha256": digest(selector/"selector_protocol.json"),
        "coverage_dev_summary_sha256": digest(selector/"evaluation/summary.json"),
        "original_frozen_sha256": digest(source/"frozen_protocol.json"),
        "code_hashes": {k: digest(p) for k, p in paths.items()}, "fixed_tuned_source_method": fixed,
    }
    return protocol, {"tools": tools, "paths": paths}, provenance


def prepare_fresh_test(model: Any, tokenizer: Any, source_experiment_dir: str | Path,
                       n_documents: int = 20, coverage_selector_dir: str | Path | None = None,
                       additional_exclusion_dirs: tuple[str | Path, ...] = ()) -> str:
    """CPU/input-only preparation; fixes choices and article list BEFORE test generation.

    Repeating the same call returns the existing frozen experiment. It does NOT
    silently sample another test set. Do not change n_documents after viewing scores.
    """
    require(type(n_documents) is int and 2 <= n_documents <= 60, "Use 2..60 articles, chosen before viewing test scores.")
    source = Path(source_experiment_dir).expanduser().resolve()
    require((source/"protocol.json").is_file(), f"Original experiment missing: {source}")
    project = source.parent.parent
    extra = tuple(Path(p).resolve() for p in additional_exclusion_dirs)
    selector = discover_selector(source, coverage_selector_dir)
    original, bundle, provenance = _source_bundle(source, selector, model, tokenizer)
    tools, paths = bundle["tools"], bundle["paths"]
    plan = {
        "version": VERSION, "runner_sha256": digest(Path(__file__)), **provenance,
        "runtime": original["runtime"], "n_documents": n_documents, "split": "fresh_test",
        "eta": ETA, "n_blocks": BLOCKS, "fixed_tradeoff_lambda": COMPARISON_LAMBDA,
        "method_labels": list(METHODS), "keep_fraction": .5, "max_new_tokens": 64,
        "system_prompt": original["system_prompt"], "source_sha256": original["source_sha256"],
        "comparison_rationale": {
            "fixed_tuned": "V1 development-tuned fixed mixture; retain it even when it equals g-only.",
            "fixed_lambda_0.75": "Prespecified supplementary trade-off comparator, chosen using DEVELOPMENT only; not the original tuned winner.",
            "coverage_eta_0.95": "Chosen within the coverage family on prior development; only this eta is evaluated here.",
        },
        "candidate_lambda_grid": original["lambda_grid"],
        "sampling": "Original heldout article-pool order, excluding all recorded prior/reserved article titles; original triple builder and input eligibility. No prediction-based filtering.",
        "additional_exclusion_dirs": [str(p) for p in extra],
        "manual_semantic_review": False, "no_pretraining_noncontamination_claim": True,
        "tune_on_fresh_test": False, "physical_compaction": False, "native_kvzip_reproduction": False,
        "measurement": "First/Same/Different/Worst F1 plus EM; retain truncated/repetitive answers.",
        "bootstrap": {"unit": "article/document with both conditions kept together", "seed": BOOTSTRAP_SEED,
                      "repetitions": BOOTSTRAP_REPETITIONS, "interval": "percentile 2.5%-97.5%; descriptive, no multiplicity correction"},
    }
    root = source.parent / f"coverage_fresh_test_v1_n{n_documents}_{objhash(plan)[:12]}"
    with run_lock(source.parent/".fresh_test_preparation.lock"):
        if (root/"frozen_test.json").is_file():
            frozen = read_json(root/"frozen_test.json")
            require(frozen["plan"] == plan, "Existing fresh-test plan differs; do not overwrite.")
            _load_frozen(root, model, tokenizer, check_current_exclusions=True)
            logger.debug('%s %s', "Existing frozen fresh test reused:", root)
            return str(root)
        require(not root.exists(), f"Incomplete preparation at {root}. Preserve and inspect it instead of overwriting.")
        exclusions = known_articles(project, exclude_root=root, extra_dirs=extra)
        # Explicitly verify original dev+heldout membership is covered by the scan.
        original_titles = {r["source_article"] for split in ("development", "heldout")
                           for r in read_jsonl(source/"data"/split/"evaluation_only.jsonl")}
        require(original_titles <= set(exclusions["article_titles"]), "Old development/heldout exclusions incomplete.")
        raw_options = [project/"gstar_fullkv_pilot/data/raw/train-v1.1.json", project/"data/raw/train-v1.1.json"]
        raw = next((p for p in raw_options if p.is_file()), None)
        require(raw is not None, "Original train-v1.1.json cache missing. Keep the existing gstar_fullkv_pilot/data/raw directory; no automatic download.")
        require(digest(raw) == original["source_sha256"], "Cached dataset differs from the original.")
        payload = tools["builder"]["parse_source"](raw.read_bytes())
        dev_pool, heldout_pool = tools["builder"]["split_articles"](payload["data"])
        manifest = read_json(source/"source_split.json")
        require([a["title"] for a in dev_pool] == manifest["development_article_titles"] and
                [a["title"] for a in heldout_pool] == manifest["heldout_article_titles"], "Original article split changed.")
        blocked = set(exclusions["article_titles"])
        candidates = [a for a in heldout_pool if a["title"] not in blocked]
        packs, screening = tools["expanded"]["_sample_packs"](candidates, n_documents, "fresh_test", tokenizer, tools)
        tools["builder"]["validate_packs"](packs, {a["title"] for a in candidates}, blocked)
        assert_disjoint(packs, exclusions)
        inputs, labels = tools["expanded"]["_export_split"](packs, tools["builder"], "fresh_test")
        root.mkdir(parents=True)
        data = root/"data/fresh_test"
        save_jsonl(data/"selection_inputs.jsonl", inputs)
        save_jsonl(data/"evaluation_only.jsonl", labels)
        (data/"preview.txt").write_text(tools["expanded"]["_preview"](inputs, labels, "fresh_test"), encoding="utf-8")
        save_json(root/"screening.json", {"candidate_pool_size": len(candidates), "checks": screening,
                                         "selected_titles": [p["source_article"] for p in packs]})
        save_json(root/"exclusions.json", exclusions)
        code = root/"code"; code.mkdir()
        for key, path in paths.items():
            shutil.copy2(path, code/(key+".py"))
        shutil.copy2(__file__, code/"fresh_runner.py")
        frozen = {
            "plan": plan, "frozen_utc": stamp(), "test_answers_seen_before_freeze": False,
            "document_ids": [p["document_id"] for p in packs],
            "article_titles": [p["source_article"] for p in packs],
            "prior_article_count": len(blocked), "prior_article_overlap": 0,
            "data_hashes": {str(p.relative_to(root)): digest(p) for p in data.iterdir() if p.is_file()},
            "screening_sha256": digest(root/"screening.json"), "exclusions_sha256": digest(root/"exclusions.json"),
        }
        save_json(root/"frozen_test.json", frozen)
        save_json(root/"freeze_seal.json", {"frozen_test_sha256": digest(root/"frozen_test.json")})
    logger.debug('%s', "========== FRESH TEST PREPARED (NO MODEL INFERENCE) ==========")
    logger.info('%s', f"New articles: {n_documents} | unique QA: {3*n_documents} | overlap with prior articles: 0")
    logger.debug('%s %s', "Coverage eta=0.95; blocks=4; supplementary fixed lambda=0.75; original fixed winner:", plan["fixed_tuned_source_method"])
    logger.debug('%s', "Settings/article IDs are frozen. No test-based tuning or replacement.")
    logger.info('%s %s', "Saved under:", root)
    return str(root)


def _load_frozen(root: Path, model: Any = None, tokenizer: Any = None,
                 check_current_exclusions: bool = True) -> tuple[dict, dict, list[dict]]:
    seal = read_json(root/"freeze_seal.json")
    require(digest(root/"frozen_test.json") == seal["frozen_test_sha256"], "Frozen test was modified.")
    frozen = read_json(root/"frozen_test.json"); plan = frozen["plan"]
    require(plan["version"] == VERSION and plan["runner_sha256"] == digest(Path(__file__)), "Fresh-test runner changed; do not merge incompatible runs.")
    for name, expected in frozen["data_hashes"].items():
        p = (root/name).resolve()
        require(p.is_relative_to(root) and p.is_file() and digest(p) == expected, f"Frozen input changed: {name}")
    require(digest(root/"screening.json") == frozen["screening_sha256"] and
            digest(root/"exclusions.json") == frozen["exclusions_sha256"], "Screening/exclusion record changed.")
    tools = {}
    for key, sha in plan["code_hashes"].items():
        p = root/"code"/(key+".py")
        require(p.is_file() and digest(p) == sha, f"Code snapshot changed: {p}")
        tools[key] = runpy.run_path(str(p))
    require(tools["expanded"]["SYSTEM_PROMPT"] == plan["system_prompt"], "Prompt changed.")
    if model is not None:
        require(tools["expanded"]["runtime_fingerprint"](model, tokenizer) == plan["runtime"],
                "Runtime changed since freezing. Reload the same model/environment; do not bypass.")
    groups = tools["expanded"]["_groups"](root, "fresh_test")
    require([g["document_id"] for g in groups] == frozen["document_ids"], "Frozen document order changed.")
    require([g["cases"][0][1]["source_article"] for g in groups] == frozen["article_titles"], "Article identities changed.")
    old = read_json(root/"exclusions.json")
    require(not set(frozen["article_titles"]) & set(old["article_titles"]), "Frozen overlap detected.")
    if check_current_exclusions:
        current = known_articles(root.parent.parent, exclude_root=root,
                                 extra_dirs=tuple(Path(p) for p in plan["additional_exclusion_dirs"]))
        require(not set(frozen["article_titles"]) & set(current["article_titles"]),
                "A frozen test article is now registered in another experiment. Inspect possible test reuse.")
    return frozen, tools, groups


def build_evaluation_policies(old_bank: dict, coverage_raw: torch.Tensor,
                              document_positions: list[int], coverage_tool: dict,
                              fixed_tuned_source: str) -> tuple[dict, dict]:
    """No target/reference/performance input. Only eta=.95 receives QA evaluation.

    The unchanged development helper constructs eta candidates internally. They
    are NOT scored on fresh questions or used for new hyperparameter selection.
    """
    cov_bank = coverage_tool["build_coverage_bank"](old_bank, coverage_raw, document_positions)
    mapping = {"full_kv": "full_kv", "g_proxy_only": "g_proxy_only",
               "fixed_tuned": fixed_tuned_source, "fixed_lambda_0.75": "fixed_lambda_0.75"}
    policies = {alias: dict(old_bank["policies"][source], source_method=source)
                for alias, source in mapping.items()}
    policies["coverage_eta_0.95"] = dict(cov_bank["policies"]["coverage_eta_0.95"])
    require(list(policies) == list(METHODS), "Evaluation method list changed.")
    require(policies["coverage_eta_0.95"]["parameter"] == ETA, "Eta changed.")
    return policies, {k: v for k, v in cov_bank.items() if k != "policies"}


def checked_result(folder: Path, frozen_hash: str) -> dict | None:
    if not (folder/"done.json").is_file():
        return None
    done = read_json(folder/"done.json")
    require(done["status"] == "COMPLETE" and done["frozen_test_sha256"] == frozen_hash, "Checkpoint/protocol mismatch.")
    attempt = (folder/done["attempt"]).resolve()
    require(attempt.is_relative_to(folder.resolve()) and bool(done["file_hashes"]), "Invalid checkpoint path/hashes.")
    for name, sha in done["file_hashes"].items():
        p = (attempt/name).resolve()
        require(p.is_relative_to(attempt) and p.is_file() and digest(p) == sha, f"Checkpoint modified: {p}")
    return read_json(attempt/"result.json")


def _run_document(model: Any, tokenizer: Any, group: dict, folder: Path,
                  tools: dict, plan: dict, frozen_hash: str) -> dict:
    paired, smoke, full = tools["paired"], tools["smoke"], tools["full"]
    render = tools["expanded"]["render_prompt"]
    attempt = folder/f"attempt_{stamp()}"; attempt.mkdir(parents=True, exist_ok=False)
    rows, diagnostics, audits = [], [], []
    actual_calls = limit_hits = 0
    document, doc_id = group["document"], group["document_id"]
    try:
        layout = paired["make_layout"](tokenizer, render, document)
        p = layout["prefix_length"]
        # Recompute the original g and check coverage's g from the same replay definition.
        g, gmeta = paired["extract_g"](model, tokenizer, render, layout, 1024)
        request_ids = paired["tokenized"](tokenizer, render(tokenizer, document, paired["RECONSTRUCTION_INSTRUCTION"]))["input_ids"]
        require(torch.equal(request_ids[:, :p], layout["prefix_ids"]), "Replay prefix changed.")
        ids = torch.cat([request_ids, torch.tensor([layout["document_token_ids"]], dtype=torch.long)], dim=1)
        coverage, g_check = tools["extractor"]["collect_coverage"](
            model, ids, layout["document_positions"], p, request_ids.shape[1], BLOCKS, 1024)
        recheck = tools["extractor"]["compare_g"](g_check, g)
        require(recheck["within_tolerance"], f"g/coverage replay mismatch: {recheck}")
        banks, suffixes, raw_cases = {}, {}, {}
        # BOTH first-question selections are finalized BEFORE any evaluation query.
        for selection, _label in group["cases"]:
            cid = selection["case_id"]
            a, suffix, qpos = paired["extract_a"](model, tokenizer, render, layout, selection["first_question"], 1024)
            old_bank = paired["build_policy_bank"](a, g, layout["document_positions"], p, .5, smoke)
            policies, cov_meta = build_evaluation_policies(old_bank, coverage, layout["document_positions"],
                                                           tools["coverage"], plan["fixed_tuned_source_method"])
            bank = {"budget": old_bank["budget"], "policies": policies}
            banks[cid] = bank; suffixes[cid] = suffix.to(model.device)
            raw_cases[cid] = {"a_raw": a, "question_positions": qpos, "first_question": selection["first_question"]}
            torch.save({"case_id": cid, **bank, "coverage_candidates": cov_meta,
                        "normalized_a_free": old_bank["normalized_a_free"],
                        "normalized_g_free": old_bank["normalized_g_free"]}, attempt/f"{cid}_selection.pt")
        torch.save({"layout": layout, "g_raw_proxy": g, "g_metadata": gmeta,
                    "coverage_raw": coverage, "g_recheck": recheck, "cases": raw_cases}, attempt/"scores.pt")
        with torch.no_grad():
            prefix = layout["prefix_ids"].to(model.device)
            output = model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                           use_cache=True, logits_to_keep=1, return_dict=True)
        base = output.past_key_values; del output
        snapshot = smoke["prefix_cache_snapshot"](base)
        require(int(base.get_seq_length()) == p, "Wrong prefix cache length.")
        first_cid = group["cases"][0][0]["case_id"]
        allkeep = torch.ones_like(banks[first_cid]["policies"]["g_proxy_only"]["keep"])
        tr0, token, _ = smoke["fixed_trace"](model, suffixes[first_cid], base, None)
        tr1, _, audit = smoke["fixed_trace"](model, suffixes[first_cid], base, allkeep, token)
        equal = smoke["compare_traces"](tr0, tr1)
        require(equal["exactly_equal"], "All-keep mask changed logits.")
        audits.append({"test": "all_keep", "comparison": equal, "audit": audit})
        for selection, label in group["cases"]:
            cid = selection["case_id"]; bank = banks[cid]
            keep = bank["policies"]["coverage_eta_0.95"]["keep"]
            clean, token, audit0 = smoke["fixed_trace"](model, suffixes[cid], base, keep)
            poisoned = smoke["poison_excluded"](base, keep)
            altered, _, audit1 = smoke["fixed_trace"](model, suffixes[cid], poisoned, keep, token)
            del poisoned
            comparison = smoke["compare_traces"](clean, altered)
            require(comparison["exactly_equal"], "Excluded-KV perturbation changed logits.")
            audits.append({"case_id": cid, "test": "coverage_excluded_kv", "comparison": comparison,
                           "clean_audit": audit0, "poisoned_audit": audit1})
            require(smoke["cache_intact"](base, snapshot, p), "Audits altered original KV.")
            questions = []
            for role, key in (("first", "first_question_reference"), ("target", "target_question_reference")):
                ref = label[key]
                question_ids = paired["tokenized"](tokenizer, render(tokenizer, document, ref["question"]))["input_ids"]
                require(torch.equal(question_ids[:, :p], layout["prefix_ids"]), "QA prefix changed.")
                require(p < question_ids.shape[1] <= 1152, "Question too long; no silent truncation.")
                questions.append((role, ref, question_ids[:, p:].to(model.device)))
            memo = {}  # independent across same/different conditions
            for method, policy in bank["policies"].items():
                diagnostics.append({"document_id": doc_id, "case_id": cid, "condition": label["condition"],
                                    "method": method, "source_method": policy["source_method"],
                                    "family": policy["family"], "parameter": policy["parameter"],
                                    "mask_key": policy["mask_key"], "diagnostics": policy["diagnostics"]})
                for role, ref, suffix in questions:
                    record_id = f"{cid}/{method}/{role}"
                    key = (policy["mask_key"], tuple(suffix.cpu().flatten().tolist()))
                    reused = None
                    if key in memo:
                        reused, result = memo[key]
                    else:
                        result = smoke["generate_answer"](model, tokenizer, suffix, base, policy["keep"], 64)
                        memo[key] = (record_id, result)
                        actual_calls += 1; limit_hits += int(result["hit_token_limit"])
                    require(smoke["cache_intact"](base, snapshot, p), "Answer generation altered original KV.")
                    refs = [x["text"] for x in ref["answers"]]
                    rows.append({"record_id": record_id, "document_id": doc_id, "case_id": cid,
                                 "project_split": "fresh_test", "source_article": label["source_article"],
                                 "condition": label["condition"], "role": role, "method": method,
                                 "family": policy["family"], "parameter": policy["parameter"],
                                 "source_method": policy["source_method"], "question_id": ref["question_id"],
                                 "question": ref["question"], "references": refs,
                                 **full["score_answer"](result["answer"], refs), "mask_key": policy["mask_key"],
                                 "original_kv_unchanged": True, "prefix_tokens": p,
                                 "kept_prefix_tokens": p if policy["keep"] is None else bank["budget"]["kept_prefix_tokens_per_layer_kv_head"],
                                 "reused_from_record": reused,
                                 "answer_is_verbatim_context_substring": bool(result["answer"]) and result["answer"] in document,
                                 "answer_whitespace_words": len(result["answer"].split()), **copy.deepcopy(result)})
                    save_jsonl(attempt/"predictions.jsonl", rows)
        pairs = tools["expanded"]["_check_pairs"](rows)
        if plan["fixed_tuned_source_method"] == "g_proxy_only":
            for role in ("first", "target"):
                for cid in banks:
                    by = {r["method"]: r for r in rows if r["case_id"] == cid and r["role"] == role}
                    require(by["fixed_tuned"]["mask_key"] == by["g_proxy_only"]["mask_key"] and
                            by["fixed_tuned"]["generated_token_ids"] == by["g_proxy_only"]["generated_token_ids"],
                            "Expected fixed-tuned/g-only alias mismatch.")
        result = {"document_id": doc_id, "source_article": group["cases"][0][1]["source_article"],
                  "rows": rows, "diagnostics": diagnostics, "audits": audits, "paired_targets": pairs,
                  "g_recheck": recheck, "actual_generation_calls": actual_calls, "token_limit_hits": limit_hits}
        save_json(attempt/"result.json", result)
        save_json(attempt/"audits.json", audits)
        save_jsonl(attempt/"diagnostics.jsonl", diagnostics)
        save_json(folder/"done.json", {"status": "COMPLETE", "attempt": attempt.name,
                                      "frozen_test_sha256": frozen_hash,
                                      "file_hashes": {p.name: digest(p) for p in attempt.iterdir() if p.is_file()}})
        return result
    except BaseException as exc:
        save_json(attempt/"failure.json", {"error": str(exc), "traceback": traceback.format_exc(),
                                          "answer_records_saved": len(rows)})
        raise


def paired_bootstrap(rows: list[dict], repeats: int = BOOTSTRAP_REPETITIONS) -> dict:
    """Same sampled document indices for both methods AND both paired conditions.

    Worst is recomputed from resampled group means, NOT averaged document minima.
    No test-side fitting, winner selection, or answer rewriting.
    """
    docs = sorted({r["document_id"] for r in rows})
    require(len(docs) >= 2, "Bootstrap requires at least two documents.")
    arrays = {}
    for method in METHODS:
        values = []
        for doc in docs:
            group = [r for r in rows if r["method"] == method and r["document_id"] == doc]
            first = [r["f1"] for r in group if r["role"] == "first"]
            same = [r["f1"] for r in group if r["role"] == "target" and r["condition"] == "same"]
            different = [r["f1"] for r in group if r["role"] == "target" and r["condition"] == "different"]
            require(len(first) == 2 and len(same) == len(different) == 1, "Incomplete bootstrap pair.")
            values.append([sum(first)/2, same[0], different[0]])
        arrays[method] = np.array(values, dtype=np.float64)
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(docs), size=(repeats, len(docs)))
    names = ("first_f1", "same_f1", "different_f1", "worst_target_f1")
    def values(a):
        mean = a.mean(axis=0)
        resampled = a[idx].mean(axis=1)
        return np.r_[mean, min(mean[1], mean[2])], np.c_[resampled, np.minimum(resampled[:,1], resampled[:,2])]
    mean_new, boot_new = values(arrays["coverage_eta_0.95"])
    comparisons = {}
    for baseline in ("g_proxy_only", "fixed_tuned", "fixed_lambda_0.75"):
        mean_base, boot_base = values(arrays[baseline])
        diffs = boot_new - boot_base
        lo, hi = np.quantile(diffs, [.025, .975], axis=0)
        comparisons["coverage_eta_0.95_minus_"+baseline] = {
            name: {"difference": float(mean_new[j]-mean_base[j]), "ci95_low": float(lo[j]), "ci95_high": float(hi[j])}
            for j, name in enumerate(names)}
    return {"documents": len(docs), "resamples": repeats, "seed": BOOTSTRAP_SEED,
            "scope": "Prespecified paired article-bootstrap percentile intervals; no multiple-comparison correction; not proof of superiority.",
            "comparisons": comparisons}


def summarize_results(results: list[dict], tools: dict, frozen: dict) -> dict:
    summary = tools["expanded"]["summarize"](results, "fresh_test", frozen["plan"]["n_documents"])
    summary["selected_on_fresh_test"] = None
    summary["no_test_tuning"] = True
    summary["frozen_eta"] = ETA
    summary["frozen_fixed_tradeoff_lambda"] = COMPARISON_LAMBDA
    summary["fixed_tuned_source_method"] = frozen["plan"]["fixed_tuned_source_method"]
    summary["prior_article_overlap"] = 0
    cov_rows = [{"case_ids": sorted({r["case_id"] for r in x["diagnostics"] if r["method"] == "coverage_eta_0.95"}),
                 "diagnostics": [r for r in x["diagnostics"] if r["method"] == "coverage_eta_0.95"]} for x in results]
    summary["coverage_diagnostics"] = tools["coverage"]["_selection_summary"](cov_rows)["methods"]
    summary["all_g_rechecks_pass"] = all(x["g_recheck"]["within_tolerance"] for x in results)
    summary["scope"] = "Fresh article-disjoint confirmation of a development-chosen coverage candidate; NO test-side selection. Masking, not physical KV compaction."
    if summary["status"] == "COMPLETE":
        summary["paired_bootstrap"] = paired_bootstrap([r for x in results for r in x["rows"]])
    return summary


def _write_results(out: Path, summary: dict, results: list[dict]) -> None:
    save_json(out/"summary.json", summary)
    save_jsonl(out/"predictions.jsonl", [r for x in results for r in x["rows"]])
    save_jsonl(out/"diagnostics.jsonl", [r for x in results for r in x["diagnostics"]])
    save_json(out/"paired_targets.json", summary["paired_targets"])
    if "paired_bootstrap" in summary:
        save_json(out/"paired_bootstrap.json", summary["paired_bootstrap"])
    with (out/"comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "source_method", "first_f1", "same_f1", "different_f1", "worst_f1", "first_em", "same_em", "different_em"])
        for name, m in summary["methods"].items():
            w.writerow([name, m["source_method"], m["first"]["f1"], m["same"]["f1"], m["different"]["f1"],
                        m["worst_target_f1"], m["first"]["em"], m["same"]["em"], m["different"]["em"]])


def run_fresh_test(model: Any, tokenizer: Any, test_dir: str | Path,
                   max_new_documents: int | None = None) -> dict:
    """Run only frozen methods. Resume skips hash-verified completed documents.

    max_new_documents limits work per invocation, NOT final sample size. Partial
    output must not be used to change eta, comparators, generation, or data.
    """
    require(max_new_documents is None or (type(max_new_documents) is int and max_new_documents > 0), "Invalid invocation document limit.")
    root = Path(test_dir).expanduser().resolve()
    with run_lock(root/".run.lock"):
        frozen, tools, groups = _load_frozen(root, model, tokenizer)
        plan = frozen["plan"]; frozen_hash = digest(root/"frozen_test.json")
        start = root/"test_started.json"
        if start.is_file():
            require(read_json(start)["frozen_test_sha256"] == frozen_hash, "Protocol changed after test started.")
        else:
            save_json(start, {"started_utc": stamp(), "frozen_test_sha256": frozen_hash})
        out = root/"evaluation"; out.mkdir(exist_ok=True)
        smoke = tools["smoke"]
        old_backend, old_training = model.config._attn_implementation, model.training
        old_hooks = smoke["hook_counts"](model)
        require(all(x == (0,0) for x in old_hooks), "Existing attention hooks. Do not run another model task in parallel.")
        results, newly = [], 0
        summary = None
        try:
            model.eval(); model.set_attn_implementation("eager")
            for i, group in enumerate(groups, 1):
                folder = out/group["document_id"]
                previous = checked_result(folder, frozen_hash)
                if previous is not None:
                    results.append(previous)
                    logger.info('%s', f"[{i}/{len(groups)}] {group['document_id']}: verified reuse; no GPU work.")
                    continue
                if max_new_documents is not None and newly >= max_new_documents:
                    continue
                logger.debug('%s', f"[{i}/{len(groups)}] {group['document_id']}: new scores, coverage, masks, QA ...")
                result = _run_document(model, tokenizer, group, folder, tools, plan, frozen_hash)
                results.append(result); newly += 1
                # Store partial results, but do not suggest test-side best parameters.
                save_jsonl(out/"predictions.jsonl", [r for x in results for r in x["rows"]])
                save_json(out/"progress.json", {"completed_this_scan": len(results), "planned": len(groups),
                                               "last_document": group["document_id"], "no_test_tuning": True})
                logger.info('%s', f"  SAVED: {group['document_id']} | audits PASS | records={len(result['rows'])}")
            require(bool(results), "No complete document yet; inspect failure logs.")
            summary = summarize_results(results, tools, frozen)
            _write_results(out, summary, results)
        except BaseException as exc:
            save_json(out/f"failure_{stamp()}.json", {"error": str(exc), "traceback": traceback.format_exc(),
                                                      "completed_this_scan": len(results)})
            logger.error('%s', "Stopped; completed checkpoints retained. Fix the runtime error, then repeat the SAME call.")
            raise
        finally:
            try:
                model.set_attn_implementation(old_backend)
            finally:
                model.train(old_training)
                restored = {"backend_after": model.config._attn_implementation,
                            "attention_backend_restored": model.config._attn_implementation == old_backend,
                            "training_mode_restored": model.training == old_training,
                            "hook_counts_restored": smoke["hook_counts"](model) == old_hooks,
                            "frozen_protocol_unchanged": digest(root/"frozen_test.json") == frozen_hash}
                save_json(out/"restoration.json", restored)
            require(all(v for k,v in restored.items() if k != "backend_after"), "Runtime settings not restored.")
        summary["restoration"] = restored; summary["output_directory"] = str(out)
        save_json(out/"summary.json", summary)
        logger.debug('%s', "\n========== FRESH COVERAGE TEST SUMMARY ==========")
        logger.debug('%s', f"Status: {summary['status']} | documents: {len(results)}/{len(groups)} | unique questions: {summary['unique_questions']}")
        if summary["status"] == "COMPLETE":
            logger.debug('%s', f"{'Method':26s} {'First':>8s} {'Same':>8s} {'Diff':>8s} {'Worst':>8s}")
            for name, m in summary["methods"].items():
                logger.debug('%s', f"{name:26s} {m['first']['f1']:8.2f} {m['same']['f1']:8.2f} {m['different']['f1']:8.2f} {m['worst_target_f1']:8.2f}")
        else:
            logger.debug('%s', "Partial checkpoint only. Continue the same frozen run; do not retune from partial scores.")
        logger.debug('%s', "Prior-article overlap: 0 | eta=0.95 | supplementary lambda=0.75")
        logger.debug('%s %s %s', "Original dev-tuned fixed:", plan["fixed_tuned_source_method"], "(not a second independent result if identical to g-only)")
        logger.debug('%s %s %s %s', "Actual QA calls:", summary["actual_generation_calls"], "| token-limit hits:", summary["token_limit_hits"])
        logger.debug('%s %s %s %s', "Backend:", restored["backend_after"], "| hooks restored:", restored["hook_counts_restored"])
        logger.debug('%s', "MASKING ONLY; short-document reconstruction proxy; no native KVzip/memory/speed claim.")
        logger.info('%s %s', "Results:", out)
        return summary


def export_review(test_dir: str | Path) -> str:
    """Export inputs, frozen settings, answers, diagnostics and audit JSON; no tensors."""
    root = Path(test_dir).expanduser().resolve()
    with run_lock(root/".run.lock"):
        _load_frozen(root, check_current_exclusions=False)
        out = root/"evaluation"
        require((out/"summary.json").is_file(), "No run summary yet.")
        archive = root/f"fresh_coverage_review_{stamp()}.zip"
        wanted = [root/n for n in ("frozen_test.json", "freeze_seal.json", "test_started.json", "exclusions.json", "screening.json")]
        wanted += list((root/"data/fresh_test").glob("*.jsonl"))
        wanted += [out/n for n in ("summary.json", "predictions.jsonl", "diagnostics.jsonl", "paired_targets.json",
                                   "paired_bootstrap.json", "comparison.csv", "restoration.json", "progress.json")]
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as z:
            for path in wanted:
                if path.is_file():
                    z.write(path, arcname=str(path.relative_to(root)))
            for marker in sorted(out.glob("fresh_test_*/done.json")):
                done = read_json(marker)
                audit = marker.parent/done["attempt"]/"audits.json"
                if audit.is_file():
                    z.write(audit, arcname=f"audits/{marker.parent.name}.json")
    logger.info('%s %s', "Review ZIP:", archive)
    return str(archive)
