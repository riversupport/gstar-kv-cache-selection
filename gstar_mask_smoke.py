"""GSTAR: one-document, layer/KV-head-wise MASKED selection smoke test.

Notebook entry point (uses an already-loaded Qwen2.5 model):
    run_mask_smoke(model, tokenizer, '/mnt/gstar-project')

Target API: transformers 4.56.x, Qwen2ForCausalLM, batch size 1,
non-sliding DynamicCache. No installation, training, or model download.

IMPORTANT
- This is access masking, NOT physical KV compaction or a speed benchmark.
- The budget counts ALL persistent prefix positions, including protected ones.
- The existing g_raw_probe is a reconstruction-attention proxy, NOT native KVzip.
- Only one existing first-question case is tested, not the paired shift experiment.
- Evaluation references never enter the scorer, selector, or model prompt.

Primary API references:
https://raw.githubusercontent.com/huggingface/transformers/v4.56.0/src/transformers/models/qwen2/modeling_qwen2.py
https://raw.githubusercontent.com/huggingface/transformers/v4.56.0/src/transformers/cache_utils.py
https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import runpy
import shutil
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import logging

logger = logging.getLogger("gstar")

VERSION = '1.0'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linked_file(value: str, fallback_dir: Path) -> Path:
    """Resolve a recorded Modal path; only its exact basename is used as fallback."""
    p = Path(value)
    if p.is_file():
        return p
    q = fallback_dir / p.name
    require(q.is_file(), f'Recorded file not found: {p}\nFallback path not found: {q}')
    return q


def make_masks(a: torch.Tensor, g: torch.Tensor, doc_positions: list[int],
               prefix_length: int, keep_fraction: float, recent_doc_tokens: int = 16,
               sink_tokens: int = 4) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Normalize on UNPROTECTED document positions, then select independently per KV head."""
    require(0 < keep_fraction < 1, 'keep_fraction must be strictly between 0 and 1.')
    require(a.ndim == 3 and a.shape == g.shape, 'a/g shape mismatch.')
    require(a.shape[-1] == len(doc_positions), 'Document score axis mismatch.')
    require(doc_positions == sorted(set(doc_positions)), 'Document positions are not unique/sorted.')
    require(doc_positions and 0 <= doc_positions[0] <= doc_positions[-1] < prefix_length,
            'Document positions are outside the persistent prefix.')
    require(recent_doc_tokens >= 0 and sink_tokens >= 0, 'Invalid protected-position counts.')
    for name, scores in [('a', a), ('g_proxy', g)]:
        require(bool(torch.isfinite(scores).all() and (scores >= 0).all()), f'Invalid {name} scores.')

    doc_idx = torch.tensor(doc_positions, dtype=torch.long)
    protected = torch.ones(prefix_length, dtype=torch.bool)
    protected[doc_idx] = False  # All non-document system/template tokens remain protected.
    protected[:min(sink_tokens, prefix_length)] = True
    if recent_doc_tokens:
        protected[doc_idx[-min(recent_doc_tokens, len(doc_idx)):]] = True
    free_doc = ~protected[doc_idx]
    free_prefix = doc_idx[free_doc]
    budget = math.floor(prefix_length * keep_fraction)
    n_protected = int(protected.sum())
    n_free_keep = budget - n_protected
    require(0 < n_free_keep < len(free_prefix),
            f'Budget {budget} vs protected {n_protected}: no nontrivial selection possible. '
            'Do not silently change the budget or protection rule.')

    masks = {}
    for name, raw in [('a_only', a), ('g_proxy_only', g)]:
        values = raw[..., free_doc].float().cpu()
        mass = values.sum(dim=-1, keepdim=True)
        require(bool((mass > 0).all()), f'{name}: an unprotected score mass is zero.')
        normalized = values / mass
        # Stable sorting makes lower absolute token position the deterministic tie-break.
        ranked = torch.argsort(normalized, dim=-1, descending=True, stable=True)
        chosen = free_prefix[ranked[..., :n_free_keep]]
        keep = protected.view(1, 1, -1).expand(a.shape[0], a.shape[1], -1).clone()
        keep.scatter_(2, chosen, True)
        require(bool((keep.sum(-1) == budget).all()), 'Budget is not exact in every layer/head.')
        require(bool(keep[..., protected].all()), 'A protected position was removed.')
        masks[name] = keep

    overlap = (masks['a_only'] & masks['g_proxy_only']).sum(-1).float()
    metadata = {
        'budget_denominator': 'all persistent prefix tokens, INCLUDING protected positions',
        'keep_fraction_requested': keep_fraction,
        'prefix_tokens': prefix_length,
        'document_tokens': len(doc_positions),
        'non_document_prefix_tokens': prefix_length - len(doc_positions),
        'protected_tokens': n_protected,
        'protected_prefix_positions': torch.where(protected)[0].tolist(),
        'protection_rule': (f'all non-document prefix tokens + first {sink_tokens} prefix tokens '
                            f'+ last {recent_doc_tokens} document tokens'),
        'recent_document_tokens_config': recent_doc_tokens,
        'sink_prefix_tokens_config': sink_tokens,
        'kept_prefix_tokens_per_layer_kv_head': budget,
        'kept_document_tokens_per_layer_kv_head': budget - (prefix_length - len(doc_positions)),
        'free_slots_per_layer_kv_head': n_free_keep,
        'achieved_prefix_fraction': budget / prefix_length,
        'mean_a_g_kept_set_overlap_fraction': float((overlap / budget).mean()),
        'tie_break': 'smaller original prefix token position',
        'scoring_scope': 'single document, single precomputed first-question case',
    }
    return masks, metadata


class PrefixMaskAudit:
    """Temporarily add per-layer/per-KV-head exclusions to Qwen2's causal mask.

    The forward hooks inspect ACTUAL eager attention weights, including every decode
    call. No source files or model weights are changed. Only prefix KEY access is
    restricted; current question/answer tokens are governed by the original causal mask.
    """
    def __init__(self, model: Any, keep: torch.Tensor):
        self.model = model
        self.keep = keep.to(device=model.device, dtype=torch.bool)
        self.prefix_length = keep.shape[-1]
        self.hq = model.config.num_attention_heads
        self.hkv = model.config.num_key_value_heads
        self.modules = [layer.self_attn for layer in model.model.layers]
        require(self.hq % self.hkv == 0, 'Invalid GQA head grouping.')
        require(tuple(keep.shape[:2]) == (len(self.modules), self.hkv), 'Mask head shape mismatch.')
        self.keep_q = self.keep.repeat_interleave(self.hq // self.hkv, dim=1)
        self.handles: list[Any] = []
        self.calls = [0] * len(self.modules)
        self.pending: dict[int, tuple[torch.Tensor, str]] = {}
        self.max_blocked = {phase: torch.zeros((), device=model.device) for phase in ('question', 'decode')}
        self.finite = torch.ones((), device=model.device, dtype=torch.bool)
        self.min_weight = torch.zeros((), device=model.device)
        self.max_rowsum_error = torch.zeros((), device=model.device)

    def __enter__(self):
        require(self.model.config._attn_implementation == 'eager', 'Audit requires eager attention.')
        try:
            for index, module in enumerate(self.modules):
                require(module.layer_idx == index, 'Unexpected attention layer ordering.')
                self.handles.append(module.register_forward_pre_hook(self._pre, with_kwargs=True))
                self.handles.append(module.register_forward_hook(self._post, with_kwargs=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()
        return False

    def _pre(self, module: Any, args: tuple, kwargs: dict):
        require(not args, 'Unexpected positional attention arguments; inspect Qwen implementation.')
        i = module.layer_idx
        hidden = kwargs['hidden_states']
        require(hidden.shape[0] == 1, 'This smoke test only supports batch size 1.')
        cp = kwargs.get('cache_position')
        require(cp is not None and cp.ndim == 1, 'Explicit cache_position is required.')
        require(bool((cp >= self.prefix_length).all()),
                'Mask attached during document prefill; prefix must be built WITHOUT selection masking.')
        qlen = hidden.shape[1]
        require(cp.numel() == qlen, 'Query/cache_position mismatch.')
        mask = kwargs.get('attention_mask')
        require(mask is not None and mask.ndim == 4 and mask.dtype.is_floating_point,
                'Expected the 4D additive eager causal mask; stop rather than guess.')
        require(mask.shape[-2] == qlen and mask.shape[1] in (1, self.hq), 'Causal-mask shape mismatch.')
        klen = mask.shape[-1]
        require(klen == int(cp[-1]) + 1, 'Only an unpadded, nonsliding DynamicCache is supported.')
        require(klen >= self.prefix_length, 'Prefix keys are missing.')
        blocked = torch.zeros((self.hq, klen), dtype=torch.bool, device=hidden.device)
        blocked[:, :self.prefix_length] = ~self.keep_q[i]
        combined = mask.expand(1, self.hq, qlen, klen).clone()
        combined.masked_fill_(blocked[None, :, None, :], float('-inf'))
        phase = 'question' if self.calls[i] == 0 else 'decode'
        self.pending[i] = (blocked, phase)
        new_kwargs = dict(kwargs)
        new_kwargs['attention_mask'] = combined
        return args, new_kwargs

    def _post(self, module: Any, args: tuple, kwargs: dict, output: Any):
        i = module.layer_idx
        require(isinstance(output, tuple) and len(output) >= 2, 'Unexpected attention output.')
        weights = output[1]
        require(isinstance(weights, torch.Tensor) and weights.ndim == 4,
                'Eager attention weights unavailable; cannot claim masking was checked.')
        blocked, phase = self.pending.pop(i)
        require(weights.shape[1] == self.hq and weights.shape[-1] == blocked.shape[-1],
                'Attention audit shape mismatch.')
        with torch.no_grad():
            self.finite &= torch.isfinite(weights).all()
            self.min_weight = torch.minimum(self.min_weight, weights.amin().float())
            self.max_rowsum_error = torch.maximum(
                self.max_rowsum_error, (weights.float().sum(-1) - 1).abs().amax())
            leakage = weights.masked_select(blocked[None, :, None, :]).abs()
            if leakage.numel():
                self.max_blocked[phase] = torch.maximum(self.max_blocked[phase], leakage.amax().float())
        self.calls[i] += 1
        # Returning None preserves the original attention output.

    def report(self, require_decode: bool = False) -> dict[str, Any]:
        equal_coverage = bool(self.calls) and min(self.calls) >= 1 and len(set(self.calls)) == 1
        decode_seen = equal_coverage and min(self.calls) >= 2
        result = {
            'all_layers_audited': equal_coverage,
            'decode_seen_in_all_layers': decode_seen,
            'calls_per_layer': self.calls.copy(),
            'max_blocked_attention_question': float(self.max_blocked['question']),
            'max_blocked_attention_decode': float(self.max_blocked['decode']) if decode_seen else None,
            'all_attention_finite': bool(self.finite),
            'min_attention_weight': float(self.min_weight),
            'max_attention_rowsum_error': float(self.max_rowsum_error),
        }
        require(equal_coverage, 'Some attention layers were not audited.')
        require(result['all_attention_finite'] and result['min_attention_weight'] >= 0,
                'Nonfinite or negative attention values.')
        require(result['max_attention_rowsum_error'] < 0.02, 'Invalid attention row sums.')
        require(result['max_blocked_attention_question'] == 0.0, 'Question processing accessed excluded KV.')
        if decode_seen:
            require(result['max_blocked_attention_decode'] == 0.0, 'Decoding accessed excluded KV.')
        if require_decode:
            require(decode_seen, 'The diagnostic did not cover a decoding step.')
        return result


def prefix_cache_snapshot(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    require(hasattr(cache, 'layers') and cache.layers, 'Expected a populated DynamicCache.')
    result = []
    for layer in cache.layers:
        require(not getattr(layer, 'is_sliding', False), 'Sliding-window caches are not supported here.')
        require(layer.keys is not None and layer.values is not None, 'Uninitialized cache layer.')
        result.append((layer.keys.detach().cpu().clone(), layer.values.detach().cpu().clone()))
    return result


def cache_intact(cache: Any, snapshot: list, prefix_length: int) -> bool:
    return (len(cache.layers) == len(snapshot) and int(cache.get_seq_length()) == prefix_length
            and all(torch.equal(layer.keys.detach().cpu(), keys)
                    and torch.equal(layer.values.detach().cpu(), values)
                    for layer, (keys, values) in zip(cache.layers, snapshot)))


def model_step(model: Any, ids: torch.Tensor, cache: Any):
    """Only new tokens are forwarded. Original, uncompressed positional indices stay intact."""
    n_past = int(cache.get_seq_length())
    end = n_past + ids.shape[1]
    return model(input_ids=ids,
                 attention_mask=torch.ones((1, end), dtype=torch.long, device=ids.device),
                 cache_position=torch.arange(n_past, end, device=ids.device),
                 past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)


@torch.no_grad()
def fixed_trace(model: Any, suffix: torch.Tensor, base_cache: Any,
                keep: torch.Tensor | None, next_token: int | None = None):
    """Question suffix + one fixed continuation input: always covers both phases.

    This is an implementation audit, not an evaluated extra answer. The continuation
    token is taken from the clean run, never from the reference answer.
    """
    work = copy.deepcopy(base_cache)
    context = PrefixMaskAudit(model, keep) if keep is not None else nullcontext(None)
    with context as audit:
        output = model_step(model, suffix, work)
        logits0 = output.logits[0, -1].float().cpu().clone()
        require(bool(torch.isfinite(logits0).all()), 'Nonfinite question logits.')
        if next_token is None:
            next_token = int(logits0.argmax())
        output = model_step(model, torch.tensor([[next_token]], device=model.device), output.past_key_values)
        logits1 = output.logits[0, -1].float().cpu().clone()
        require(bool(torch.isfinite(logits1).all()), 'Nonfinite continuation logits.')
        report = audit.report(require_decode=True) if audit is not None else None
    return [logits0, logits1], next_token, report


def compare_traces(left: list, right: list) -> dict[str, Any]:
    return {'exactly_equal': all(torch.equal(x, y) for x, y in zip(left, right)),
            'max_abs_logit_difference': max(float((x - y).abs().max()) for x, y in zip(left, right))}


def poison_excluded(base_cache: Any, keep: torch.Tensor) -> Any:
    """Modify ONLY excluded KV on an independent copy. Never poison retained KV."""
    poisoned = copy.deepcopy(base_cache)
    changed = False
    for i, layer in enumerate(poisoned.layers):
        block = (~keep[i]).to(layer.keys.device)[None, :, :, None]
        require(layer.keys.shape[-2] == block.shape[-2], 'Poisoning needs the pristine prefix cache.')
        changed |= bool((layer.values.masked_select(block) != 17.0).any())
        layer.keys.masked_fill_(block, 0.0)
        layer.values.masked_fill_(block, 17.0)  # finite; NaN/inf would invalidate this diagnostic.
    require(changed, 'Poison test failed to change any excluded values.')
    return poisoned


@torch.no_grad()
def generate_answer(model: Any, tokenizer: Any, suffix: torch.Tensor, base_cache: Any,
                    keep: torch.Tensor | None, max_new_tokens: int) -> dict[str, Any]:
    """Manual greedy loop: no full-prompt re-prefill inside an evaluated answer."""
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    work = copy.deepcopy(base_cache)
    generated: list[int] = []
    context = PrefixMaskAudit(model, keep) if keep is not None else nullcontext(None)
    started = time.perf_counter()
    with context as audit:
        current = suffix
        for _ in range(max_new_tokens):
            output = model_step(model, current, work)
            work = output.past_key_values
            scores = output.logits[0, -1]
            require(bool(torch.isfinite(scores).all()), 'Nonfinite output logits.')
            token = int(scores.argmax())
            generated.append(token)
            if token in eos_ids:
                break
            current = torch.tensor([[token]], dtype=torch.long, device=model.device)
        report = audit.report() if audit is not None else None
    return {
        'answer': tokenizer.decode(generated, skip_special_tokens=True).strip(),
        'generated_token_ids': generated,
        'hit_token_limit': len(generated) == max_new_tokens and generated[-1] not in eos_ids,
        'mask_audit': report,
        'seconds_including_audits_not_a_speed_benchmark': time.perf_counter() - started,
    }


def hook_counts(model: Any) -> list[tuple[int, int]]:
    return [(len(layer.self_attn._forward_pre_hooks), len(layer.self_attn._forward_hooks))
            for layer in model.model.layers]


def run_mask_smoke(model: Any, tokenizer: Any,
                   project_dir: str | Path = '/mnt/gstar-project',
                   keep_fraction: float = 0.5, max_new_tokens: int = 64,
                   g_probe_path: str | Path | None = None) -> dict[str, Any]:
    """Load linked score files, validate masking, then answer two questions with 3 methods."""
    import transformers

    require(transformers.__version__.startswith('4.56.'),
            f'Expected Transformers 4.56.x; found {transformers.__version__}. Do not reinstall automatically.')
    require(model.config.model_type == 'qwen2', 'This runner is specific to Qwen2/Qwen2.5.')
    require(hasattr(model, 'set_attn_implementation'), 'Attention backend switching is unavailable.')
    require(not getattr(model, 'is_quantized', False), 'Use the already-loaded, unquantized BF16 model.')
    require(1 <= max_new_tokens <= 128, 'Use max_new_tokens between 1 and 128 for this smoke test.')
    project = Path(project_dir).expanduser().resolve()
    require(project.is_dir(), f'Project Volume missing: {project}')
    results_dir = project / 'results'
    runner_path = project / 'gstar_fullkv_pilot' / 'gstar_run_fullkv.py'
    require(runner_path.is_file(), f'Existing Full-KV runner missing: {runner_path}')
    runner = runpy.run_path(str(runner_path))
    render = runner['render_prompt']

    # Select g first, and follow its recorded a-file link (never pair independent latest files).
    if g_probe_path is None:
        candidates = sorted(results_dir.glob('attention_g_probe_*.pt'))
        require(bool(candidates), 'No attention_g_probe_*.pt file found. Keep the existing score files.')
        g_path = candidates[-1]
    else:
        g_path = Path(g_probe_path)
        require(g_path.is_file(), f'g probe missing: {g_path}')
    g_obj = torch.load(g_path, map_location='cpu', weights_only=True)
    a_path = linked_file(g_obj['source_a_probe'], results_dir)
    a_obj = torch.load(a_path, map_location='cpu', weights_only=True)
    require(g_obj.get('native_kvzip_reproduction') is False, 'Unexpected g definition; inspect metadata.')
    for obj in (a_obj, g_obj):
        require(obj.get('normalized') is False and obj.get('protected_positions_applied') is False,
                'Expected the raw, unprotected probe scores.')
        require(obj.get('scoring_backend') == 'eager', 'Expected eager scoring probes.')
        saved_id = obj.get('model_id')
        current_id = model.config._name_or_path
        require(not saved_id or saved_id == current_id, 'Probe/model ID mismatch.')
        old_rev, current_rev = obj.get('model_revision'), getattr(model.config, '_commit_hash', None)
        require(not (old_rev and current_rev) or old_rev == current_rev, 'Probe/model revision mismatch.')
    require(a_obj['document_id'] == g_obj['document_id'], 'The a/g documents differ.')
    require(a_obj['document_positions'] == g_obj['document_positions'], 'The a/g token positions differ.')
    require(a_obj['document_token_ids'] == g_obj['document_token_ids'], 'The a/g document tokens differ.')

    env_path = Path(a_obj['source_environment'])
    if not env_path.is_file():
        matches = list(results_dir.glob(f'{env_path.parent.name}/environment.json'))
        require(len(matches) == 1, f'Linked environment file missing: {env_path}')
        env_path = matches[0]
    env = json.loads(env_path.read_text(encoding='utf-8'))
    # Prefer the frozen snapshot from the exact original Full-KV run.
    data_dir = env_path.parent / 'data_snapshot'
    if not (data_dir / 'selection_inputs.jsonl').is_file():
        data_dir = Path(env['data_directory'])
    require((data_dir / 'selection_inputs.jsonl').is_file(), f'Selection data missing: {data_dir}')
    selection_rows = read_jsonl(data_dir / 'selection_inputs.jsonl')
    cases = [r for r in selection_rows if r['case_id'] == a_obj['case_id']]
    require(len(cases) == 1, 'Missing/duplicate score case in selection inputs.')
    case = cases[0]
    require(case['document_id'] == a_obj['document_id'] and case['first_question'] == a_obj['first_question'],
            'Case/probe identity mismatch.')
    doc, first_question = case['document'], case['first_question']
    marker = '__GSTAR_QUESTION_SLOT__'
    marked = render(tokenizer, doc, marker)
    require(marked.count(marker) == 1, 'Question marker is not unique.')
    prefix_text = marked.split(marker)[0]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False, return_tensors='pt')['input_ids']
    prefix_length = prefix_ids.shape[1]
    doc_positions = a_obj['document_positions']
    a, g = a_obj['a_raw'].float(), g_obj['g_raw_probe'].float()
    require(tuple(a.shape[:2]) == (model.config.num_hidden_layers, model.config.num_key_value_heads),
            'Score/model layer or head mismatch.')
    require(prefix_length <= 1024, 'This runner is a short-document smoke test (prefix <= 1024).')
    require(prefix_ids[0, doc_positions].tolist() == a_obj['document_token_ids'],
            'Tokenized document changed; do not apply saved scores to a different prompt.')
    require(int(g_obj['query_start']) == prefix_length, 'g reconstruction prefix length changed.')
    require(torch.equal(g_obj['input_ids'][:, :prefix_length], prefix_ids), 'g reconstruction prefix changed.')

    masks, budget_info = make_masks(a, g, doc_positions, prefix_length, keep_fraction)
    all_keep = torch.ones_like(masks['a_only'])

    # ONLY NOW load gold labels/target question: they cannot influence mask construction.
    labels = [r for r in read_jsonl(data_dir / 'evaluation_only.jsonl') if r['case_id'] == case['case_id']]
    require(len(labels) == 1, 'Missing/duplicate evaluation case.')
    label = labels[0]
    require(label['project_split'] == 'development', 'This is a DEVELOPMENT smoke test only.')
    require(label['first_question_reference']['question'] == first_question, 'First-question label mismatch.')
    questions = [('first', label['first_question_reference']), ('target', label['target_question_reference'])]
    encoded_questions = []
    for role, ref in questions:
        ids = tokenizer(render(tokenizer, doc, ref['question']), add_special_tokens=False,
                        return_tensors='pt')['input_ids']
        require(torch.equal(ids[:, :prefix_length], prefix_ids), 'Question tokenization changed the prefix.')
        require(prefix_length < ids.shape[1] <= 1152, 'Unexpected question input length.')
        for answer in ref['answers']:
            start, text = answer['document_answer_start'], answer['text']
            require(doc[start:start + len(text)] == text, 'Gold answer span is not in the document.')
        encoded_questions.append((role, ref, ids[:, prefix_length:].to(model.device)))

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out_dir = results_dir / f'mask_smoke_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out_dir / 'runner_snapshot.py')
    torch.save({'masks': masks, 'budget': budget_info, 'document_positions': doc_positions,
                'prefix_input_ids': prefix_ids, 'case_id': case['case_id']}, out_dir / 'selection.pt')
    metadata = {
        'runner_version': VERSION, 'model_id': model.config._name_or_path,
        'model_revision': getattr(model.config, '_commit_hash', None),
        'model_dtype': str(model.dtype), 'torch_version': str(torch.__version__),
        'transformers_version': transformers.__version__, 'device': str(model.device),
        'evaluation_backend': 'eager FOR ALL methods including Full-KV',
        'case_id': case['case_id'], 'document_id': case['document_id'],
        'condition_of_this_single_case': label['condition'],
        'a_probe': str(a_path), 'a_sha256': sha256(a_path),
        'g_probe': str(g_path), 'g_sha256': sha256(g_path),
        'g_definition': g_obj.get('score_definition'), 'native_kvzip_reproduction': False,
        'source_environment': str(env_path), 'source_data_directory': str(data_dir),
        'fullkv_runner_sha256': sha256(runner_path), 'system_prompt': runner.get('SYSTEM_PROMPT'),
        'max_new_tokens': max_new_tokens, 'decoding': 'manual greedy; no sampling or logits processors',
        'physical_compaction': False, 'hardware_memory_savings_measured': False,
        'new_question_answer_tokens_are_not_in_persistent_prefix_budget': True,
        'budget': budget_info,
    }
    write_json(out_dir / 'environment.json', metadata)
    logger.debug('%s', '========== ONE-DOCUMENT MASK SMOKE TEST ==========')
    logger.debug('%s', f"Case: {case['case_id']} | condition: {label['condition']} (NOT paired evaluation)")
    logger.debug('%s', f'Score files: {a_path.name} + {g_path.name}')
    logger.debug('%s', f"Prefix: {prefix_length} | protected: {budget_info['protected_tokens']} | "
          f"kept: {budget_info['kept_prefix_tokens_per_layer_kv_head']} "
          f"({budget_info['achieved_prefix_fraction']:.2%}) per layer/KV-head")
    logger.debug('%s', f'Run directory: {out_dir}')

    old_backend = model.config._attn_implementation
    old_training = model.training
    before_hooks = hook_counts(model)
    summary: dict[str, Any] = {}
    try:
        model.eval()
        model.set_attn_implementation('eager')
        require(model.config._attn_implementation == 'eager', 'Could not switch to eager attention.')
        with torch.no_grad():
            prefill = model(input_ids=prefix_ids.to(model.device),
                            attention_mask=torch.ones_like(prefix_ids, device=model.device),
                            use_cache=True, logits_to_keep=1, return_dict=True)
        base_cache = prefill.past_key_values
        del prefill
        snapshot = prefix_cache_snapshot(base_cache)
        require(len(base_cache.layers) == model.config.num_hidden_layers, 'Cache layer count mismatch.')
        require(int(base_cache.get_seq_length()) == prefix_length, 'Cache/prefix length mismatch.')

        logger.debug('%s', '[1/3] Full-KV vs all-keep mask: question + continuation logits ...')
        suffix = encoded_questions[0][2]
        baseline_trace, next_token, _ = fixed_trace(model, suffix, base_cache, None)
        all_trace, _, all_report = fixed_trace(model, suffix, base_cache, all_keep, next_token)
        all_match = compare_traces(baseline_trace, all_trace)
        write_json(out_dir / 'all_keep_check.json', {'comparison': all_match, 'audit': all_report})
        require(all_match['exactly_equal'], 'All-keep mask changed logits. Inspect all_keep_check.json.')
        logger.debug('%s', '  All-keep equivalence: PASS')

        logger.debug('%s', '[2/3] Excluded-KV perturbation + actual attention audits ...')
        poison_checks = {}
        for name, keep in masks.items():
            clean_trace, continuation, clean_report = fixed_trace(model, suffix, base_cache, keep)
            poisoned = poison_excluded(base_cache, keep)
            poison_trace, _, poison_report = fixed_trace(model, suffix, poisoned, keep, continuation)
            del poisoned
            comparison = compare_traces(clean_trace, poison_trace)
            poison_checks[name] = {'comparison': comparison, 'clean_audit': clean_report,
                                   'perturbed_audit': poison_report}
            write_json(out_dir / 'excluded_kv_checks.json', poison_checks)
            require(comparison['exactly_equal'],
                    f'{name}: excluded KV perturbation changed logits. Inspect excluded_kv_checks.json.')
            require(cache_intact(base_cache, snapshot, prefix_length), 'Original prefix cache was modified.')
            logger.debug('%s', f'  {name}: PASS | blocked attention 0 (question AND continuation) | '
                  'perturbation logit difference 0')

        logger.debug('%s', '[3/3] Answer generation: Full-KV / a-only / g-proxy-only ...')
        rows = []
        methods = [('full_kv', None), *masks.items()]
        with (out_dir / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
            for name, keep in methods:
                for role, ref, suffix in encoded_questions:
                    result = generate_answer(model, tokenizer, suffix, base_cache, keep, max_new_tokens)
                    refs = [x['text'] for x in ref['answers']]
                    metrics = runner['score_answer'](result['answer'], refs)
                    intact = cache_intact(base_cache, snapshot, prefix_length)
                    require(intact, 'Original prefix cache changed after answer generation.')
                    row = {'method': name, 'case_id': case['case_id'], 'document_id': case['document_id'],
                           'condition': label['condition'], 'role': role, 'question_id': ref['question_id'],
                           'question': ref['question'], 'references': refs, 'em': metrics['em'],
                           'f1': metrics['f1'], 'original_kv_unchanged': intact,
                           'kept_prefix_tokens': prefix_length if keep is None else budget_info['kept_prefix_tokens_per_layer_kv_head'],
                           'prefix_tokens': prefix_length, **result}
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    handle.flush()
                    rows.append(row)
                    logger.debug('%s', f"  {name:13s} | {role:6s} | EM={row['em']:6.2f} F1={row['f1']:6.2f} | {row['answer']!r}")
        summary = {
            'status': 'PASS', 'case_id': case['case_id'], 'documents': 1, 'unique_questions': 2,
            'evaluated_answers': len(rows), 'all_keep_equivalence': all_match,
            'excluded_kv_checks': poison_checks,
            'original_kv_unchanged': all(r['original_kv_unchanged'] for r in rows),
            'generation_token_limit_hits': sum(r['hit_token_limit'] for r in rows),
            'budget': budget_info,
            'scores': [{'method': r['method'], 'role': r['role'], 'em': r['em'], 'f1': r['f1'],
                        'answer': r['answer']} for r in rows],
            'scope': 'ONE CASE smoke test, not a paired evidence-shift result; no tuning or test-set evaluation',
            'native_kvzip_reproduction': False, 'physical_compaction': False,
            'output_directory': str(out_dir),
        }
    except BaseException as exc:
        write_json(out_dir / 'failure.json', {'status': 'FAILED', 'error': str(exc),
                                             'traceback': traceback.format_exc()})
        logger.error('%s', f'Audit stopped. Existing results are preserved. Error log: {out_dir / "failure.json"}')
        raise
    finally:
        # Context managers remove their own hooks even when a forward fails.
        model.set_attn_implementation(old_backend)
        model.train(old_training)
        restored = {'attention_backend_restored': model.config._attn_implementation == old_backend,
                    'backend_after': model.config._attn_implementation,
                    'hook_counts_restored': hook_counts(model) == before_hooks}
        write_json(out_dir / 'restoration.json', restored)
        if summary:
            summary.update(restored)
            require(restored['attention_backend_restored'] and restored['hook_counts_restored'],
                    'Model settings were not completely restored; inspect restoration.json.')
            write_json(out_dir / 'summary.json', summary)

    logger.debug('%s', '\n========== MASK SMOKE SUMMARY ==========')
    logger.debug('%s', 'All-keep equivalence: PASS')
    logger.debug('%s', 'Excluded attention = 0 in question + continuation: PASS')
    logger.debug('%s', 'Excluded-KV perturbation invariance: PASS')
    logger.debug('%s', 'Original KV values unchanged: True')
    logger.debug('%s %s', 'Hooks removed:', summary['hook_counts_restored'])
    logger.debug('%s %s', 'Attention backend restored:', summary['backend_after'])
    logger.debug('%s %s', 'Generation token-limit hits:', summary['generation_token_limit_hits'])
    logger.info('%s %s', 'Results saved:', out_dir)
    logger.debug('%s', 'MASKING ONLY: no physical KV compaction, no hardware-savings claim.')
    return summary
