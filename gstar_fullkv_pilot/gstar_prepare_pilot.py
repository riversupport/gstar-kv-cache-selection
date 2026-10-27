# -*- coding: utf-8 -*-
"""GStar: prepare a SMALL, DEVELOPMENT-ONLY paired QA pilot on the CPU.

Usage: python gstar_prepare_pilot.py
Requires Python 3.8+ and internet for the initial download; no extra packages or GPU.
Does not modify or import gstar_day1_selector.py.

1. Download the original SQuAD 1.1 training file; no LLM training is performed.
2. Split source articles into project-specific development and held-out pools first.
3. Build three small, two-passage documents from the development pool.
4. Hold each document and target question fixed while varying the first question.
5. Store reference answers, targets, and evidence labels separately from selection inputs.

This script only prepares data; it does not compute attention, a/g scores, or
LLM answers. The two-passage pilot is not a 2k-4k-token benchmark.
The same/different labels use the paragraph containing the annotated answer.
Cross-passage answer-string checks do not establish unique semantic evidence;
human review is still needed. Source passages, questions, and answers are unchanged.

Data: Rajpurkar et al., SQuAD 1.1 (CC BY-SA 4.0).
https://rajpurkar.github.io/SQuAD-explorer/
https://creativecommons.org/licenses/by-sa/4.0/
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import logging

logger = logging.getLogger("gstar")

# Fixed pilot configuration; change only in a separately versioned experiment.
PILOT_PACKS = 3
SPLIT_SEED = 42
DEV_FRACTION = 0.25   # Project-specific split; the official SQuAD splits are unchanged.
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
DATA_URLS = (
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json",
)
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "train-v1.1.json"
OUT_DIR = ROOT / "data" / "pilot"
Json = Dict[str, Any]


def require(condition: bool, message: str) -> None:
    """Raise an explicit error when a validation check fails."""
    if not condition:
        raise ValueError(message)


def digest_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_source(payload: bytes) -> Json:
    obj = json.loads(payload.decode("utf-8"))
    require(isinstance(obj, dict), "The downloaded file is not a JSON object.")
    require(str(obj.get("version")) == "1.1", "Expected SQuAD version 1.1.")
    require(isinstance(obj.get("data"), list) and bool(obj["data"]),
            "The source has no articles.")
    return obj


def load_source() -> Tuple[Json, str]:
    """Cache the source data without disabling TLS verification or installing packages."""
    if RAW_PATH.exists():
        logger.debug('%s', "Using cached source: data/raw/train-v1.1.json")
        payload = RAW_PATH.read_bytes()
        return parse_source(payload), hashlib.sha256(payload).hexdigest()

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for index, url in enumerate(DATA_URLS, 1):
        logger.debug('%s', "Downloading SQuAD 1.1 (about 30 MB), source %d/%d ..." %
              (index, len(DATA_URLS)))
        try:
            request = Request(url, headers={"User-Agent": "GStar-Pilot/1.0"})
            with urlopen(request, timeout=60) as response:
                payload = response.read(MAX_DOWNLOAD_BYTES + 1)
            require(len(payload) <= MAX_DOWNLOAD_BYTES, "Unexpectedly large download.")
            obj = parse_source(payload)  # Do not cache HTML error pages as dataset files.
            temp_path = RAW_PATH.with_suffix(".json.part")
            temp_path.write_bytes(payload)
            temp_path.replace(RAW_PATH)
            logger.debug('%s', "Download complete: %.1f MB" % (len(payload) / 1000000))
            return obj, hashlib.sha256(payload).hexdigest()
        except (HTTPError, URLError, OSError, ValueError) as exc:
            errors.append("Source %d: %s" % (index, exc))
    raise RuntimeError("Could not download the source. No packages need reinstalling.\n"
                       + "\n".join(errors)
                       + "\nCopy this terminal output and ask for help.")


def split_articles(articles: List[Json]) -> Tuple[List[Json], List[Json]]:
    """Split source articles before selecting paragraphs or questions."""
    titles = [a.get("title") for a in articles]
    require(all(isinstance(t, str) and t for t in titles), "Invalid article title.")
    require(len(set(titles)) == len(titles), "Duplicate article titles need grouping first.")
    require(len(articles) >= 2, "At least two articles are required.")
    ranked = sorted(articles, key=lambda a: digest_key(
        str(SPLIT_SEED) + "|article|" + a["title"]))
    n_dev = max(1, int(len(ranked) * DEV_FRACTION))
    return ranked[:n_dev], ranked[n_dev:]


def valid_questions(paragraph: Json) -> List[Json]:
    """Keep only questions whose annotated answer spans match the source text."""
    context = paragraph.get("context", "")
    valid = []
    if not isinstance(context, str) or not context:
        return valid
    for qa in paragraph.get("qas", []):
        answers = qa.get("answers", [])
        if not qa.get("id") or not isinstance(qa.get("question"), str) or not answers:
            continue
        spans_ok = True
        for answer in answers:
            start, text = answer.get("answer_start"), answer.get("text")
            if (type(start) is not int or start < 0 or not isinstance(text, str)
                    or not text or context[start:start + len(text)] != text):
                spans_ok = False
                break
        if spans_ok:
            valid.append(qa)
    return valid


def disjoint_answers(left: Json, right: Json) -> bool:
    """Exclude duplicate or overlapping answers within the same paragraph."""
    for a in left["answers"]:
        for b in right["answers"]:
            if a["text"].strip().casefold() == b["text"].strip().casefold():
                return False
            a0, b0 = a["answer_start"], b["answer_start"]
            if max(a0, b0) < min(a0 + len(a["text"]), b0 + len(b["text"])):
                return False
    return True


def absent_from(qa: Json, text: str) -> bool:
    """Check string overlap only; this does not establish semantic independence."""
    return all(a["text"].casefold() not in text.casefold() for a in qa["answers"])


def choose_triple(article: Json) -> Optional[Tuple[int, int, Json, Json, Json]]:
    """Select the target and same-first question from one passage and different-first from another."""
    paragraphs = article.get("paragraphs", [])
    eligible = []
    for pi, paragraph in enumerate(paragraphs):
        # Short, readable pilot: these are whitespace word counts, not model token counts.
        words = len(paragraph.get("context", "").split())
        if 40 <= words <= 220:
            qas = valid_questions(paragraph)
            if qas:
                eligible.append((pi, paragraph["context"], qas))
    for ti, t_context, t_qas in eligible:
        for target in t_qas:
            for same in t_qas:
                if (target["id"] == same["id"]
                        or target["question"].casefold() == same["question"].casefold()
                        or not disjoint_answers(target, same)):
                    continue
                for di, d_context, d_qas in eligible:
                    if (di == ti or not absent_from(target, d_context)
                            or not absent_from(same, d_context)):
                        continue
                    for different in d_qas:
                        questions = {q["question"].strip().casefold()
                                     for q in (target, same, different)}
                        if len(questions) == 3 and absent_from(different, t_context):
                            return ti, di, target, same, different
    return None


def mapped_question(qa: Json, source_pi: int, passages: List[Json]) -> Json:
    passage = next(p for p in passages if p["source_paragraph_index"] == source_pi)
    return {
        "question_id": qa["id"],
        "question": qa["question"],
        "evidence_passage": passage["passage_number"],
        "answers": [{
            "text": a["text"],
            "source_answer_start": a["answer_start"],
            "document_answer_start": passage["document_start"] + a["answer_start"],
        } for a in qa["answers"]],
    }


def make_pack(article: Json, triple: Tuple[int, int, Json, Json, Json], number: int) -> Json:
    ti, di, target, same, different = triple
    order = [ti, di]
    # Use the same deterministic, seeded passage order in both conditions.
    if int(digest_key(str(SPLIT_SEED) + "|order|" + article["title"]), 16) % 2:
        order.reverse()
    document = ""
    passages = []
    for passage_number, source_pi in enumerate(order, 1):
        if document:
            document += "\n\n"
        document += "[Passage %d]\n" % passage_number
        start = len(document)
        context = article["paragraphs"][source_pi]["context"]
        document += context
        passages.append({"passage_number": passage_number,
                         "source_paragraph_index": source_pi,
                         "document_start": start, "document_end": len(document)})
    return {
        "document_id": "pilot_%03d" % number,
        "project_split": "development",
        "source_article": article["title"],
        "document": document,
        "passages": passages,
        "target": mapped_question(target, ti, passages),
        "same_first": mapped_question(same, ti, passages),
        "different_first": mapped_question(different, di, passages),
    }


def build_packs(dev_articles: List[Json], count: int) -> List[Json]:
    packs = []
    for article in dev_articles:
        triple = choose_triple(article)
        if triple is not None:
            packs.append(make_pack(article, triple, len(packs) + 1))
        if len(packs) == count:
            break
    require(len(packs) == count, "Not enough eligible DEVELOPMENT articles for the pilot.")
    return packs


def validate_packs(packs: List[Json], dev_titles: set, heldout_titles: set) -> None:
    require(not dev_titles.intersection(heldout_titles), "Article split overlap.")
    seen_articles = set()
    for pack in packs:
        title = pack["source_article"]
        require(title in dev_titles and title not in heldout_titles, "Non-dev pilot article.")
        require(title not in seen_articles, "Pilot must use distinct source articles.")
        seen_articles.add(title)
        document = pack["document"]
        target, same, different = pack["target"], pack["same_first"], pack["different_first"]
        require(len({q["question_id"] for q in (target, same, different)}) == 3,
                "Repeated question ID in a pair.")
        require(target["evidence_passage"] == same["evidence_passage"], "Invalid same label.")
        require(target["evidence_passage"] != different["evidence_passage"], "Invalid different label.")
        for q in (target, same, different):
            passage = next(p for p in pack["passages"]
                           if p["passage_number"] == q["evidence_passage"])
            for answer in q["answers"]:
                start, text = answer["document_answer_start"], answer["text"]
                require(document[start:start + len(text)] == text, "Broken answer offset.")
                require(passage["document_start"] <= start and
                        start + len(text) <= passage["document_end"], "Answer outside passage.")


def export_rows(packs: List[Json]) -> Tuple[List[Json], List[Json]]:
    selection_inputs, evaluation_only = [], []
    for pack in packs:
        for suffix, condition, first_key in (("A", "same", "same_first"),
                                             ("B", "different", "different_first")):
            first = pack[first_key]
            case_id = pack["document_id"] + "_" + suffix
            # Only document and first_question from this dictionary are selector inputs.
            selection_inputs.append({"case_id": case_id, "document_id": pack["document_id"],
                                     "document": pack["document"], "first_question": first["question"]})
            evaluation_only.append({"case_id": case_id, "document_id": pack["document_id"],
                                    "project_split": "development", "condition": condition,
                                    "source_article": pack["source_article"],
                                    "passages": pack["passages"],
                                    "first_question_reference": first,
                                    "target_question_reference": pack["target"],
                                    "manual_review_status": "not_reviewed"})
    for i in range(0, len(selection_inputs), 2):
        require(selection_inputs[i]["document"] == selection_inputs[i + 1]["document"],
                "Paired documents differ.")
        require(evaluation_only[i]["target_question_reference"] ==
                evaluation_only[i + 1]["target_question_reference"], "Paired targets differ.")
        require(selection_inputs[i]["first_question"] != selection_inputs[i + 1]["first_question"],
                "First questions must differ.")
    allowed_keys = {"case_id", "document_id", "document", "first_question"}
    require(all(set(row) == allowed_keys for row in selection_inputs), "Unexpected selection field.")
    return selection_inputs, evaluation_only


def preview_text(packs: List[Json]) -> str:
    lines = ["GSTAR DEVELOPMENT PILOT - HUMAN REVIEW ONLY",
             "Real SQuAD passages/questions; NOT model-generated answers.",
             "Short two-passage pilot, NOT the final 2k-4k-token benchmark.",
             "Reference answers and evidence labels below MUST NOT be fed to cache selection.",
             "Check manually that each question is clear and uses its indicated passage.",
             "String/offset validation does NOT guarantee unique semantic evidence.",
             "Source: SQuAD 1.1, Rajpurkar et al. (2016); data license: CC BY-SA 4.0.",
             "https://rajpurkar.github.io/SQuAD-explorer/",
             "https://creativecommons.org/licenses/by-sa/4.0/", ""]
    for pack in packs:
        lines += ["=" * 72, "DOCUMENT: " + pack["document_id"],
                  "SOURCE ARTICLE: " + pack["source_article"], "",
                  pack["document"], "", "--- Questions and reference answers ---"]
        for label, key in (("TARGET (fixed in both cases)", "target"),
                           ("FIRST / SAME annotated passage", "same_first"),
                           ("FIRST / DIFFERENT annotated passage", "different_first")):
            q = pack[key]
            lines += [label, "  Q: " + q["question"],
                      "  Reference: " + " | ".join(a["text"] for a in q["answers"]),
                      "  Annotated evidence: Passage %d" % q["evidence_passage"], ""]
        lines += ["Manual review: NOT YET REVIEWED", ""]
    return "\n".join(lines) + "\n"


def save_outputs(packs: List[Json], dev: List[Json], heldout: List[Json], source_sha: str) -> None:
    inputs, labels = export_rows(packs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Development-only short pilot. Not official SQuAD evaluation.",
        "source_dataset": "SQuAD 1.1", "source_original_split": "train",
        "source_urls": list(DATA_URLS), "source_sha256": source_sha,
        "data_license": "CC BY-SA 4.0", "seed": SPLIT_SEED,
        "article_split_rule": "Sort SHA256(seed|article|title); first floor(25%) is project development.",
        "development_article_titles": [a["title"] for a in dev],
        "heldout_article_titles": [a["title"] for a in heldout],
        "pilot_article_titles": [p["source_article"] for p in packs],
        "heldout_examples_exported": 0,
        "tokenizer_used": None,
        "sampling_note": "One pair per eligible article; paragraphs 40-220 whitespace words."
                         " Disjoint same-passage answer spans; cross-passage answer-string exclusion."
                         " Original contexts/questions unchanged; passage headers added and spans remapped.",
    }
    # Re-running overwrites generated files; preserve manual edits under different names.
    for name, content in (("selection_inputs.jsonl", inputs), ("evaluation_only.jsonl", labels)):
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in content)
        (OUT_DIR / name).write_text(text, encoding="utf-8")
    (OUT_DIR / "source_split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "preview.txt").write_text(preview_text(packs), encoding="utf-8")


def self_test() -> None:
    """Test data transformation and validation with synthetic documents and no download."""
    text_a = ("Maple Library opened in 2008. Its front door opens at nine each morning. "
              "The building contains a quiet comfortable reading room beside the entrance. Visitors "
              "can borrow books, use public desks, and attend local educational workshops throughout the entire year.")
    text_b = ("Harbor Museum was founded in 2016. Its main doors open at ten each morning. "
              "The building displays paintings and historical objects in several galleries. "
              "Visitors can explore exhibitions, join guided tours, and learn about the surrounding city and its many neighborhoods.")
    def qa(qid: str, question: str, answer: str, text: str) -> Json:
        return {"id": qid, "question": question,
                "answers": [{"text": answer, "answer_start": text.index(answer)}]}
    article = {"title": "Fictional self-test only", "paragraphs": [
        {"context": text_a, "qas": [
            qa("q1", "When did Maple Library open?", "2008", text_a),
            qa("q2", "At what hour does Maple Library open?", "nine", text_a)]},
        {"context": text_b, "qas": [
            qa("q3", "When was Harbor Museum founded?", "2016", text_b)]}]}
    triple = choose_triple(article)
    require(triple is not None, "Self-test: could not construct a pair.")
    packs = [make_pack(article, triple, 1)]
    validate_packs(packs, {article["title"]}, {"Heldout"})
    inputs, labels = export_rows(packs)
    require(len(inputs) == 2 and len(labels) == 2, "Self-test: incorrect case count.")
    require("target_question" not in inputs[0], "Self-test: target leak.")
    pool = [{"title": "Fixture %02d" % i} for i in range(12)]
    dev1, heldout1 = split_articles(pool)
    dev2, heldout2 = split_articles(list(reversed(pool)))
    require([a["title"] for a in dev1] == [a["title"] for a in dev2], "Unstable split.")
    require(len(dev1) == 3 and len(heldout1) == 9 and len(heldout2) == 9, "Bad split size.")
    broken = json.loads(json.dumps(packs))
    broken[0]["target"]["answers"][0]["document_answer_start"] += 1
    try:
        validate_packs(broken, {article["title"]}, {"Heldout"})
    except ValueError:
        pass
    else:
        raise ValueError("Self-test: a broken answer offset was not detected.")


def main() -> None:
    logger.debug('%s', "GStar CPU-only data preparation. No LLM / no GPU / no extra packages.")
    self_test()
    logger.info('%s', "SELF-TEST: PASS (data builder only)")
    if "--self-test" in sys.argv[1:]:
        return
    source, source_sha = load_source()
    dev, heldout = split_articles(source["data"])
    packs = build_packs(dev, PILOT_PACKS)
    validate_packs(packs, {a["title"] for a in dev}, {a["title"] for a in heldout})
    save_outputs(packs, dev, heldout, source_sha)
    logger.info('%s', "VALIDATION: PASS (structure and answer offsets)")
    logger.debug('%s', "Development pilot documents: %d" % len(packs))
    logger.debug('%s', "Paired selection cases: %d" % (2 * len(packs)))
    logger.debug('%s', "Held-out test examples exported: 0")
    logger.debug('%s', "No model answers or accuracy scores have been computed.")
    logger.debug('%s', "Open next: data/pilot/preview.txt")
    logger.debug('%s', "Check DOCUMENT pilot_001 first. Reference answers are for review only.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        logger.error('%s', "\nERROR: %s" % error)
        sys.exit(1)
