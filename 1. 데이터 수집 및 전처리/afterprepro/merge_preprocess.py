#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_sejong_dataset.py  (PII 마스킹/안전필터 제거판)
- perplexity 세종.txt 변환 → perplexity 세종.json
- sejong_QA_p1~39 병합 → gpt 세종.json
- claude + gemini + perplexity + gpt 4개 병합
- 정제/중복제거 후 final_dataset.json 생성(분할 없음)
"""

import os, re, json, glob, unicodedata, hashlib, argparse
from datetime import datetime
from collections import Counter

# sklearn이 있으면 TF-IDF 근접중복 제거 사용
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------- 공통 유틸 ----------------
def _normalize_text(s: str, keep_case=False) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    if not keep_case:
        s = s.lower()
    return s

def _approx_tokens(s: str) -> int:
    if not s:
        return 0
    return max(len(re.findall(r"\w+|[^\w\s]", s, re.UNICODE)), 1)

REQ_KEYS = ["instruction", "input", "output"]

def _uid(rec: dict) -> str:
    base = unicodedata.normalize("NFKC", (rec.get("instruction", "") + "||" + rec.get("input", "")).strip())
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _read_json_any(path: str) -> list[dict]:
    """JSON 배열 또는 JSONL 모두 지원."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
    # JSON array
    try:
        obj = json.loads(data)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass
    # JSONL
    items = []
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
            if isinstance(x, dict):
                items.append(x)
        except Exception:
            continue
    return items

def _write_json_array(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

# ---------------- 1) Perplexity TXT → JSON ----------------
def convert_perplexity_txt_to_json(txt_path: str, out_json_path: str) -> int:
    """TXT에서 최상위 { ... } 객체들을 추출해서 JSON 배열로 저장."""
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = unicodedata.normalize("NFKC", raw).strip()

    objs, buf, depth, in_obj = [], [], 0, False
    for ch in raw:
        if ch == "{":
            depth += 1
            in_obj = True
        if in_obj:
            buf.append(ch)
        if ch == "}":
            depth -= 1
            if in_obj and depth == 0:
                obj_text = "".join(buf).strip()
                obj_text = re.sub(r",\s*$", "", obj_text)  # 객체 뒤 꼬리콤마 제거
                try:
                    objs.append(json.loads(obj_text))
                except Exception:
                    try:
                        fixed = re.sub(r",(\s*[}\]])", r"\1", obj_text)
                        objs.append(json.loads(fixed))
                    except Exception:
                        pass
                buf, in_obj = [], False

    _write_json_array(out_json_path, objs)
    return len(objs)

# ---------------- 2) sejong_QA_p1~39 병합 → gpt 세종.json ----------------
def merge_sejong_parts_to_json(glob_pattern: str, out_json_path: str) -> int:
    """sejong_QA_p*.txt들에서 JSON 객체를 추출해 하나의 JSON 배열로 저장."""
    paths = sorted(glob.glob(glob_pattern))
    all_objs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
        raw = unicodedata.normalize("NFKC", raw).strip()
        buf, depth, in_obj = [], 0, False
        for ch in raw:
            if ch == "{":
                depth += 1
                in_obj = True
            if in_obj:
                buf.append(ch)
            if ch == "}":
                depth -= 1
                if in_obj and depth == 0:
                    obj_text = "".join(buf).strip()
                    obj_text = re.sub(r",\s*$", "", obj_text)
                    try:
                        all_objs.append(json.loads(obj_text))
                    except Exception:
                        try:
                            fixed = re.sub(r",(\s*[}\]])", r"\1", obj_text)
                            all_objs.append(json.loads(fixed))
                        except Exception:
                            pass
                    buf, in_obj = [], False
    _write_json_array(out_json_path, all_objs)
    return len(all_objs)

# ---------------- 3) 4개 데이터 전체 병합 ----------------
def merge_four_sources(claude_json: str, gemini_json: str, perplexity_json: str, gpt_json: str) -> list[dict]:
    a = _read_json_any(claude_json)
    b = _read_json_any(gemini_json)
    c = _read_json_any(perplexity_json)
    d = _read_json_any(gpt_json)
    return a + b + c + d

# ---------------- 4) 정제/중복제거(분할 없음) ----------------
def clean_and_dedup(records: list[dict], lang: str = "ko", max_tokens: int = 512,
                    dedup_tfidf: float = 0.9) -> tuple[list[dict], dict]:
    report = {
        "count_raw": len(records),
        "invalid": Counter(),
        "removed_exact_dups": 0,
        "removed_near_dups": 0,
        "near_dup_avg_compares_per_item": 0.0,
        "count_after_clean": 0,
        "sklearn_available": SKLEARN_AVAILABLE,
    }

    # 4-1) 정규화/검증
    clean = []
    for rec in records:
        if not isinstance(rec, dict):
            report["invalid"]["not_a_dict"] += 1
            continue
        if not all(k in rec for k in REQ_KEYS):
            report["invalid"]["missing_required_keys"] += 1
            continue

        rec["instruction"] = _normalize_text(rec.get("instruction", ""))
        rec["input"] = _normalize_text(rec.get("input", ""))
        rec["output"] = _normalize_text(rec.get("output", ""), keep_case=True)
        rec["lang"] = lang

        if _approx_tokens(rec["instruction"]) < 3:
            report["invalid"]["instruction_too_short"] += 1
            continue
        out_tok = _approx_tokens(rec["output"])
        if out_tok < 1 or out_tok > max_tokens:
            report["invalid"]["output_len_invalid"] += 1
            continue

        rec["uid"] = _uid(rec)
        rec.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
        clean.append(rec)

    # 4-2) 정확 중복 제거
    seen = set()
    unique = []
    for r in clean:
        if r["uid"] in seen:
            report["removed_exact_dups"] += 1
            continue
        seen.add(r["uid"])
        unique.append(r)

    # 4-3) 근접 중복 제거(TF-IDF)
    if SKLEARN_AVAILABLE and dedup_tfidf is not None and dedup_tfidf >= 0:
        texts = [r.get("instruction", "") + " " + r.get("input", "") for r in unique]
        try:
            vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
            X = vec.fit_transform(texts)
            kept, kept_idx, removed, comps = [], [], 0, 0
            for i in range(X.shape[0]):
                x = X[i]
                drop = False
                if kept_idx:
                    sub = X[kept_idx]
                    sim = (x @ sub.T).toarray().ravel()
                    comps += sim.size
                    if sim.max(initial=0.0) >= dedup_tfidf:
                        drop = True
                if drop:
                    removed += 1
                else:
                    kept.append(unique[i])
                    kept_idx.append(i)
            unique = kept
            report["removed_near_dups"] = removed
            report["near_dup_avg_compares_per_item"] = round(comps / max(1, len(texts)), 3)
        except Exception:
            pass

    report["count_after_clean"] = len(unique)
    return unique, report

# ---------------- 파이프라인 실행 ----------------
def build_final_dataset(
    base_dir: str = "preprocessing/dataset_sejong",
    claude_name: str = "claude_data.json",
    gemini_name: str = "gemini_data(400,250,125,130,115).json",
    perplexity_txt_name: str = "perplexity 세종.txt",
    sejong_glob_pattern: str = "sejong_QA_p*.txt",
    perplexity_json_name: str = "perplexity 세종.json",
    gpt_json_name: str = "gpt 세종.json",
    final_name: str = "final_dataset.json",
    report_name: str = "report.json",
    lang: str = "ko",
    max_tokens: int = 512,
    dedup_tfidf: float = 0.9,
):
    os.makedirs(base_dir, exist_ok=True)

    claude_path = os.path.join(base_dir, claude_name)
    gemini_path = os.path.join(base_dir, gemini_name)
    perplexity_txt_path = os.path.join(base_dir, perplexity_txt_name)
    sejong_glob = os.path.join(base_dir, sejong_glob_pattern)

    perplexity_json_path = os.path.join(base_dir, perplexity_json_name)
    gpt_json_path = os.path.join(base_dir, gpt_json_name)
    final_path = os.path.join(base_dir, final_name)
    report_path = os.path.join(base_dir, report_name)

    # 1) perplexity txt → json
    n_perp = convert_perplexity_txt_to_json(perplexity_txt_path, perplexity_json_path)

    # 2) sejong parts 병합 → gpt 세종.json
    n_gpt = merge_sejong_parts_to_json(sejong_glob, gpt_json_path)

    # 3) 4개 합치기  ★ 변경: 각 소스 리스트를 따로 읽어 개수 기록
    claude_items = _read_json_any(claude_path)
    gemini_items = _read_json_any(gemini_path)
    perp_items   = _read_json_any(perplexity_json_path)  # 방금 변환한 JSON
    gpt_items    = _read_json_any(gpt_json_path)         # 방금 병합한 JSON

    merged = claude_items + gemini_items + perp_items + gpt_items

    # 4) 정제/중복제거
    cleaned, rep = clean_and_dedup(merged, lang=lang, max_tokens=max_tokens, dedup_tfidf=dedup_tfidf)

    # 5) 최종 저장 + 리포트  ★ 변경: counts에 claude/gemini 추가
    _write_json_array(final_path, cleaned)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_dir": base_dir,
            "counts": {
                "perplexity_parsed": n_perp,
                "gpt_merged_from_parts": n_gpt,
                "claude": len(claude_items),          # ← 추가
                "gemini": len(gemini_items),          # ← 추가
                "raw_merged_total": len(merged),
                "final_cleaned": rep["count_after_clean"],
            },
            "invalid": rep["invalid"],
            "removed_exact_dups": rep["removed_exact_dups"],
            "removed_near_dups": rep["removed_near_dups"],
            "near_dup_avg_compares_per_item": rep["near_dup_avg_compares_per_item"],
            "sklearn_available": rep["sklearn_available"],
            "params": {"lang": lang, "max_tokens": max_tokens, "dedup_tfidf": dedup_tfidf},
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] final → {final_path} (items={rep['count_after_clean']})")
    print(f"[REPORT] {report_path}")
    return final_path

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="preprocessing/dataset_sejong")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--dedup_tfidf", type=float, default=0.9)
    args = ap.parse_args()

    build_final_dataset(
        base_dir=args.base_dir,
        lang=args.lang,
        max_tokens=args.max_tokens,
        dedup_tfidf=args.dedup_tfidf,
    )
