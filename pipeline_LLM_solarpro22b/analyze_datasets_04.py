# -*- coding: utf-8 -*-
"""
python analyze_datasets_03.py --train data/train.csv --dev data/dev.csv --morpheme_backend kiwi

Analyze dialogue→summary datasets for ROUGE-oriented evaluation.
"""
import argparse
import os
import re
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Morpheme Analyzer
# --------------------------

_MORPHEME_ANALYZER = None
_ANALYZER_BACKEND = None


def _get_morpheme_analyzer(prefer: str = "auto"):
    """형태소 분석기 초기화"""
    global _MORPHEME_ANALYZER, _ANALYZER_BACKEND

    if _MORPHEME_ANALYZER is not None:
        return _MORPHEME_ANALYZER, _ANALYZER_BACKEND

    # 1. Kiwipiepy 시도
    if prefer in ("auto", "kiwi", "kiwipiepy"):
        try:
            from kiwipiepy import Kiwi
            kiwi = Kiwi()
            print("[INFO] Using Kiwipiepy for morpheme analysis")

            class KiwiWrapper:
                def __init__(self, analyzer):
                    self.kiwi = analyzer

                def morphs(self, text):
                    """모든 형태소 추출"""
                    result = self.kiwi.tokenize(str(text))
                    return [token.form for token in result]

                def nouns(self, text):
                    """명사만 추출"""
                    result = self.kiwi.tokenize(str(text))
                    return [token.form for token in result if token.tag.startswith('N')]

            wrapper = KiwiWrapper(kiwi)
            _MORPHEME_ANALYZER = wrapper
            _ANALYZER_BACKEND = "kiwipiepy"
            return wrapper, "kiwipiepy"
        except ImportError:
            if prefer in ("kiwi", "kiwipiepy"):
                print(
                    "[WARNING] Kiwipiepy not installed. Install with: pip install kiwipiepy")
        except Exception as e:
            print(f"[WARNING] Kiwipiepy initialization failed: {e}")

    # 2. Mecab 시도
    if prefer in ("auto", "mecab"):
        try:
            from konlpy.tag import Mecab
            mecab = Mecab()
            print("[INFO] Using Mecab for morpheme analysis")
            _MORPHEME_ANALYZER = mecab
            _ANALYZER_BACKEND = "mecab"
            return mecab, "mecab"
        except Exception as e:
            if prefer == "mecab":
                print(f"[WARNING] Mecab initialization failed: {e}")

    # 3. Okt 시도
    if prefer in ("auto", "okt"):
        try:
            from konlpy.tag import Okt
            okt = Okt()
            print("[INFO] Using Okt for morpheme analysis")
            _MORPHEME_ANALYZER = okt
            _ANALYZER_BACKEND = "okt"
            return okt, "okt"
        except Exception as e:
            if prefer == "okt":
                print(f"[WARNING] Okt initialization failed: {e}")

    # 4. Fallback: 정규식 (조사 제거 없음)
    print("[WARNING] No morpheme analyzer available. Using regex (no particle removal)")

    class RegexAnalyzer:
        def morphs(self, text):
            return str(text).split()

        def nouns(self, text):
            words = re.findall(r'[가-힣]{2,}', str(text))
            return words

    analyzer = RegexAnalyzer()
    _MORPHEME_ANALYZER = analyzer
    _ANALYZER_BACKEND = "regex"
    return analyzer, "regex"


def extract_keywords(text: str, use_nouns_only: bool = False) -> set:
    """
    형태소 분석으로 키워드 추출 (조사 제거)

    Args:
        text: 입력 텍스트
        use_nouns_only: True면 명사만, False면 모든 형태소
    """
    analyzer, backend = _get_morpheme_analyzer()
    text = str(text).lower()

    # 대화 구분자 제거 (#Person1#, #Person2# 등)
    text = re.sub(r'#person\d+#', '', text, flags=re.IGNORECASE)
    text = re.sub(r'person\d+', '', text, flags=re.IGNORECASE)

    # 숫자 추출 (형태소 분석 전)
    nums = set(re.findall(r"\d[\d,.:/%]*", text))

    # 형태소 분석
    if use_nouns_only or backend == "regex":
        # 명사만 추출 또는 regex fallback
        morphemes = analyzer.nouns(text)
    else:
        # 모든 형태소 (조사 자동 분리됨)
        morphemes = analyzer.morphs(text)

    # 불용어 제거: person 관련
    stopwords = {'person', 'person1', 'person2', 'person3', 'person4'}

    # 2글자 이상만 유지, 불용어 제외
    words = set(w for w in morphemes if len(w) >=
                2 and w.lower() not in stopwords)

    return nums | words


# --------------------------
# Utility
# --------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    return list(zip(*[tokens[i:] for i in range(n)]))


def corpus_ngrams(series: pd.Series, n: int = 2, topk: int = 50000) -> Counter:
    cnt = Counter()
    for text in series.astype(str).values:
        t = normalize_text(text).split()
        cnt.update(ngrams(t, n))
    return Counter(dict(cnt.most_common(topk)))


def overlap_ratio(counter_a: Counter, counter_b: Counter) -> float:
    if not counter_a or not counter_b:
        return 0.0
    inter = sum((counter_a & counter_b).values())
    total = sum(counter_b.values())
    return inter / total if total > 0 else 0.0


def lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        xi = x[i-1]
        row = dp[i]
        prev_row = dp[i-1]
        for j in range(1, n+1):
            if xi == y[j-1]:
                row[j] = prev_row[j-1] + 1
            else:
                row[j] = max(prev_row[j], row[j-1])
    return dp[m][n]


def rouge_scores(pred: str, ref: str) -> Dict[str, float]:
    p = normalize_text(pred).split()
    r = normalize_text(ref).split()

    def f1_for_n(n: int):
        p_ngr = Counter(ngrams(p, n))
        r_ngr = Counter(ngrams(r, n))
        overlap = sum((p_ngr & r_ngr).values())
        p_count = max(1, sum(p_ngr.values()))
        r_count = max(1, sum(r_ngr.values()))
        prec = overlap / p_count
        rec = overlap / r_count
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        return prec, rec, f1

    p1, r1, f1_1 = f1_for_n(1)
    p2, r2, f1_2 = f1_for_n(2)
    l = lcs_length(p, r)
    prec_l = l / max(1, len(p))
    rec_l = l / max(1, len(r))
    f1_l = 0.0 if (prec_l + rec_l) == 0 else 2 * \
        prec_l * rec_l / (prec_l + rec_l)

    return {
        "rouge1_p": p1, "rouge1_r": r1, "rouge1_f1": f1_1,
        "rouge2_p": p2, "rouge2_r": r2, "rouge2_f1": f1_2,
        "rougeL_p": prec_l, "rougeL_r": rec_l, "rougeL_f1": f1_l,
    }


# --------------------------
# Sentence Splitter
# --------------------------

_SENTENCE_SPLITTER = None
_SPLITTER_BACKEND = None


def _get_sentence_splitter(prefer: str = "auto"):
    global _SENTENCE_SPLITTER, _SPLITTER_BACKEND

    if _SENTENCE_SPLITTER is not None:
        return _SENTENCE_SPLITTER, _SPLITTER_BACKEND

    if prefer in ("auto", "kiwi", "kiwipiepy"):
        try:
            from kiwipiepy import Kiwi
            kiwi = Kiwi()
            print("[INFO] Using kiwipiepy for sentence splitting")

            def split_fn(text: str):
                text = str(text).strip()
                if not text:
                    return []
                try:
                    result = kiwi.split_into_sents(text)
                    return [s.text.strip() for s in result if s.text.strip()]
                except Exception:
                    return [s.strip() for s in re.split(r"[.!?。\n]+", text) if s.strip()]

            _SENTENCE_SPLITTER = split_fn
            _SPLITTER_BACKEND = "kiwipiepy"
            return split_fn, "kiwipiepy"
        except:
            pass

    if prefer in ("auto", "pykss", "kss"):
        try:
            import kss
            print("[INFO] Using pykss for sentence splitting")

            def split_fn(text: str):
                text = str(text).strip()
                if not text:
                    return []
                try:
                    return [s.strip() for s in kss.split_sentences(text) if s.strip()]
                except Exception:
                    return [s.strip() for s in re.split(r"[.!?。\n]+", text) if s.strip()]

            _SENTENCE_SPLITTER = split_fn
            _SPLITTER_BACKEND = "pykss"
            return split_fn, "pykss"
        except:
            pass

    print("[INFO] Using regex for sentence splitting")

    def split_fn(text: str):
        text = str(text).strip()
        if not text:
            return []
        return [s.strip() for s in re.split(r"[.!?。\n]+", text) if s.strip()]

    _SENTENCE_SPLITTER = split_fn
    _SPLITTER_BACKEND = "regex"
    return split_fn, "regex"


def split_sentences(text: str) -> List[str]:
    splitter, _ = _get_sentence_splitter()
    return splitter(text)


def lead_k(text: str, k: int) -> str:
    parts = split_sentences(text)
    return " ".join(parts[:k])


# --------------------------
# Column inference
# --------------------------

def infer_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    target_cands = [c for c in cols if re.search(
        r"(summary|target|label|reference|abstract|gt|gold)", c, re.I)]
    source_cands = [c for c in cols if re.search(
        r"(dialog|context|text|article|content|input|document|utter|source|conversation)", c, re.I)]
    avg_len = df.apply(lambda s: s.astype(str).str.len().mean())
    sorted_by_len = avg_len.sort_values(ascending=False).index.tolist()

    source = source_cands[0] if source_cands else (
        sorted_by_len[0] if sorted_by_len else None)
    target = None
    for c in target_cands:
        if c != source:
            target = c
            break
    if target is None:
        for c in sorted_by_len:
            if c != source:
                target = c
                break
    if source is None or target is None:
        raise ValueError("Failed to infer source/target columns.")
    return source, target


# --------------------------
# Stats & visualizations
# --------------------------

def dataset_stats(df: pd.DataFrame, name: str, src: str, tgt: str) -> Dict[str, float]:
    def count_words(s): return len(str(s).split())
    def count_sents(s): return len(split_sentences(str(s)))
    stats = {
        "dataset": name,
        "rows": len(df),
        "source_col": src,
        "target_col": tgt,
        "missing_source": int(df[src].isna().sum()),
        "missing_target": int(df[tgt].isna().sum()),
        "avg_source_chars": float(df[src].astype(str).str.len().mean()),
        "avg_target_chars": float(df[tgt].astype(str).str.len().mean()),
        "avg_source_words": float(df[src].astype(str).map(count_words).mean()),
        "avg_target_words": float(df[tgt].astype(str).map(count_words).mean()),
        "avg_source_sents": float(df[src].astype(str).map(count_sents).mean()),
        "avg_target_sents": float(df[tgt].astype(str).map(count_sents).mean()),
        "unique_targets": int(df[tgt].nunique()),
    }
    return stats


def add_len_cols(df: pd.DataFrame, src: str, tgt: str) -> pd.DataFrame:
    out = df.copy()
    out["_src_words"] = out[src].astype(str).map(lambda s: len(s.split()))
    out["_tgt_words"] = out[tgt].astype(str).map(lambda s: len(s.split()))
    out["_compression"] = out["_tgt_words"] / \
        out["_src_words"].replace(0, np.nan)

    # Novelty 계산 (형태소 분석 버전)
    nov = []
    for s_src, s_tgt in zip(out[src].astype(str).values, out[tgt].astype(str).values):
        src_set = extract_keywords(s_src, use_nouns_only=False)
        tgt_list = list(extract_keywords(s_tgt, use_nouns_only=False))
        if len(tgt_list) == 0:
            nov.append(0.0)
        else:
            nov.append(
                sum([1 for w in tgt_list if w not in src_set]) / len(tgt_list))
    out["_novelty_morpheme"] = nov
    return out


def plot_dev_distributions(dev_len: pd.DataFrame, artifacts_dir: str):
    plt.figure()
    dev_len["_src_words"].dropna().astype(float).plot(
        kind="hist", bins=40, alpha=0.5, label="dev source (words)")
    dev_len["_tgt_words"].dropna().astype(float).plot(
        kind="hist", bins=40, alpha=0.5, label="dev summary (words)")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    plt.title("Dev length distribution (words)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "dev_length_hist.png"))
    plt.close()

    plt.figure()
    plt.scatter(dev_len["_src_words"], dev_len["_tgt_words"], s=8, alpha=0.5)
    plt.xlabel("Source word count")
    plt.ylabel("Summary word count")
    plt.title("Dev source vs summary length")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "dev_src_vs_tgt_scatter.png"))
    plt.close()

    plt.figure()
    dev_len["_compression"].replace([np.inf, -np.inf], np.nan).dropna().plot(
        kind="hist", bins=40, alpha=0.8)
    plt.xlabel("Compression ratio")
    plt.ylabel("Frequency")
    plt.title("Dev compression ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "dev_compression_ratio_hist.png"))
    plt.close()

    plt.figure()
    dev_len["_novelty_morpheme"].dropna().plot(kind="hist", bins=40, alpha=0.8)
    plt.xlabel("Morpheme novelty (not in source)")
    plt.ylabel("Frequency")
    plt.title("Dev summary morpheme novelty")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "dev_novelty_hist.png"))
    plt.close()


# --------------------------
# Baselines & leakage
# --------------------------

def compute_leadk_baselines(dev: pd.DataFrame, src: str, tgt: str, ks=(1, 2, 3)) -> pd.DataFrame:
    rows = []
    for k in ks:
        scores = []
        for s_src, s_ref in zip(dev[src].astype(str).values, dev[tgt].astype(str).values):
            pred = lead_k(s_src, k)
            scores.append(rouge_scores(pred, s_ref))
        agg = pd.DataFrame(scores).mean(numeric_only=True).to_dict()
        agg["k"] = k
        rows.append(agg)
    cols = ["k", "rouge1_f1", "rouge2_f1", "rougeL_f1", "rouge1_p",
            "rouge1_r", "rouge2_p", "rouge2_r", "rougeL_p", "rougeL_r"]
    out = pd.DataFrame(rows)
    return out[cols]


def error_analysis_lead3(dev: pd.DataFrame, src: str, tgt: str, topn: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    recs = []
    for i, (s_src, s_ref) in enumerate(zip(dev[src].astype(str).values, dev[tgt].astype(str).values)):
        pred = lead_k(s_src, 3)
        r = rouge_scores(pred, s_ref)
        recs.append(
            {"idx": i, "rougeL_f1": r["rougeL_f1"], "src": s_src, "ref": s_ref, "pred": pred})
    df = pd.DataFrame(recs).sort_values("rougeL_f1", ascending=False)
    return df.head(topn), df.tail(topn)


def leakage_bigram_table(train: pd.DataFrame, dev: pd.DataFrame, train_src: str, dev_src: str, dev_tgt: str) -> pd.DataFrame:
    train_src_bi = corpus_ngrams(train[train_src], n=2, topk=50000)
    dev_src_bi = corpus_ngrams(dev[dev_src], n=2, topk=50000)
    dev_tgt_bi = corpus_ngrams(dev[dev_tgt], n=2, topk=50000)
    return pd.DataFrame({
        "metric": ["train_src vs dev_src bigram overlap", "train_src vs dev_tgt bigram overlap"],
        "overlap_ratio": [
            overlap_ratio(train_src_bi, dev_src_bi),
            overlap_ratio(train_src_bi, dev_tgt_bi)
        ]
    })


def dump_top_bigrams(train: pd.DataFrame, dev: pd.DataFrame, train_src: str, dev_src: str, dev_tgt: str, artifacts_dir: str, topn: int = 100):
    def top_to_df(c: Counter, n: int) -> pd.DataFrame:
        items = c.most_common(n)
        return pd.DataFrame({"bigram": [" ".join(k) for k, _ in items], "count": [v for _, v in items]})

    top_to_df(corpus_ngrams(train[train_src], 2), topn).to_csv(
        os.path.join(artifacts_dir, "top_bigrams_train_src.csv"), index=False)
    top_to_df(corpus_ngrams(dev[dev_src], 2), topn).to_csv(
        os.path.join(artifacts_dir, "top_bigrams_dev_src.csv"), index=False)
    top_to_df(corpus_ngrams(dev[dev_tgt], 2), topn).to_csv(
        os.path.join(artifacts_dir, "top_bigrams_dev_tgt.csv"), index=False)


def bigram_overlap_table_pair(a_series, b_series, name_a, name_b, topk=50000):
    a_bi = corpus_ngrams(a_series, n=2, topk=topk)
    b_bi = corpus_ngrams(b_series, n=2, topk=topk)
    return pd.DataFrame({
        "pair": [f"{name_a} vs {name_b}"],
        "overlap_ratio_bigram": [overlap_ratio(a_bi, b_bi)]
    })


def best_support_pos(src_text: str, sum_text: str) -> List[float]:
    src_s = split_sentences(src_text)
    sum_s = split_sentences(sum_text)
    pos = []
    for s in sum_s:
        best, best_j = 0.0, 0
        for j, t in enumerate(src_s):
            sc = rouge_scores(t, s)["rougeL_f1"]
            if sc > best:
                best, best_j = sc, j
        if src_s:
            pos.append((best_j + 1) / len(src_s))
    return pos


def compute_lead_support_stats(df: pd.DataFrame, src: str, tgt: str) -> pd.DataFrame:
    all_pos = []
    for s_src, s_tgt in zip(df[src].astype(str).values, df[tgt].astype(str).values):
        all_pos += best_support_pos(s_src, s_tgt)
    if not all_pos:
        return pd.DataFrame([{"mean": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan}])
    a = pd.Series(all_pos)
    return pd.DataFrame([{
        "mean": float(a.mean()),
        "p25": float(a.quantile(0.25)),
        "p50": float(a.quantile(0.50)),
        "p75": float(a.quantile(0.75)),
        "share_front30": float((a <= 0.30).mean()),
        "share_back30": float((a >= 0.70).mean()),
        "count": int(len(a)),
    }])


def redundancy_rate(text: str) -> float:
    toks = normalize_text(text).split()
    if len(toks) < 2:
        return 0.0
    bi = Counter(ngrams(toks, 2))
    total = sum(bi.values())
    return 0.0 if total == 0 else sum(v for v in bi.values() if v > 1) / total


def entity_number_coverage(src_text: str, sum_text: str) -> dict:
    """형태소 분석 기반 키워드 커버리지 (조사 제거)"""
    src_k = extract_keywords(src_text, use_nouns_only=False)
    sum_k = extract_keywords(sum_text, use_nouns_only=False)

    if not src_k:
        return {"keyword_recall": float("nan"), "keyword_precision": float("nan")}
    rec = len(src_k & sum_k) / len(src_k)
    prec = (len(src_k & sum_k) / len(sum_k)) if sum_k else 0.0
    return {"keyword_recall": rec, "keyword_precision": prec}


def compute_quality_table(df: pd.DataFrame, src: str, tgt: str) -> pd.DataFrame:
    recs = []
    for s_src, s_tgt in zip(df[src].astype(str).values, df[tgt].astype(str).values):
        recs.append({
            "redundancy": redundancy_rate(s_tgt),
            **entity_number_coverage(s_src, s_tgt),
        })
    return pd.DataFrame(recs).mean(numeric_only=True).to_frame().T


def export_keyword_details(df: pd.DataFrame, src: str, tgt: str, output_path: str, split_name: str):
    """
    각 샘플의 키워드 추출 상세 정보를 CSV로 저장

    Args:
        df: 데이터프레임
        src: 원문 컬럼명
        tgt: 요약 컬럼명
        output_path: 저장 경로
        split_name: 분할 이름 (train/dev)
    """
    print(f"[INFO] Exporting keyword details for {split_name}...")

    records = []
    for idx, row in df.iterrows():
        s_src = str(row[src])
        s_tgt = str(row[tgt])

        # 키워드 추출 (조사 제거)
        src_keywords = extract_keywords(s_src, use_nouns_only=False)
        tgt_keywords = extract_keywords(s_tgt, use_nouns_only=False)

        # 겹치는 키워드 (교집합)
        common_keywords = src_keywords & tgt_keywords

        # 원문에만 있는 키워드 (차집합)
        src_only = src_keywords - tgt_keywords

        # 요약에만 있는 키워드 (차집합)
        tgt_only = tgt_keywords - src_keywords

        # 통계
        src_count = len(src_keywords)
        tgt_count = len(tgt_keywords)
        common_count = len(common_keywords)

        recall = common_count / src_count if src_count > 0 else 0
        precision = common_count / tgt_count if tgt_count > 0 else 0

        records.append({
            "index": idx,
            "fname": row.get("fname", f"{split_name}_{idx}"),
            "src_keyword_count": src_count,
            "tgt_keyword_count": tgt_count,
            "common_keyword_count": common_count,
            "keyword_recall": round(recall, 4),
            "keyword_precision": round(precision, 4),
            "src_keywords": "|".join(sorted(src_keywords)),
            "tgt_keywords": "|".join(sorted(tgt_keywords)),
            "common_keywords": "|".join(sorted(common_keywords)),
            "src_only_keywords": "|".join(sorted(src_only)),
            "tgt_only_keywords": "|".join(sorted(tgt_only)),
        })

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] Keyword details saved to: {output_path}")
    print(f"           Total samples: {len(records)}")

    return result_df


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True,
                    help="Path to train.csv")
    ap.add_argument("--dev", type=str, required=True, help="Path to dev.csv")
    ap.add_argument("--artifacts_dir", type=str,
                    default="artifacts", help="Output directory")
    ap.add_argument("--morpheme_backend", type=str, default="auto",
                    choices=["auto", "kiwi", "kiwipiepy",
                             "mecab", "okt", "regex"],
                    help="형태소 분석기 선택: auto(추천), mecab, okt, regex")

    args = ap.parse_args()
    ensure_dir(args.artifacts_dir)

    print(
        f"[INFO] Initializing morpheme analyzer (preference: {args.morpheme_backend})...")
    _, morph_backend = _get_morpheme_analyzer(args.morpheme_backend)
    print(f"[INFO] Using morpheme backend: {morph_backend}")

    print("[INFO] Loading datasets...")
    train = pd.read_csv(args.train)
    dev = pd.read_csv(args.dev)

    train_src, train_tgt = infer_cols(train)
    dev_src, dev_tgt = infer_cols(dev)
    print(f"[INFO] Inferred columns:")
    print(f"  train: source='{train_src}', target='{train_tgt}'")
    print(f"  dev  : source='{dev_src}', target='{dev_tgt}'")

    def schema_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
        avg_len = df.apply(lambda s: s.astype(str).str.len().mean())
        return pd.DataFrame({"dataset": name, "column": list(df.columns), "avg_char_len": np.round(avg_len.values, 1)})

    print("[INFO] Generating schema overview...")
    schema_overview = pd.concat(
        [schema_df(train, "train"), schema_df(dev, "dev")], ignore_index=True)
    schema_overview.to_csv(os.path.join(
        args.artifacts_dir, "schema_overview.csv"), index=False)

    print("[INFO] Computing dataset statistics...")
    stats_df = pd.DataFrame([
        dataset_stats(train, "train", train_src, train_tgt),
        dataset_stats(dev, "dev", dev_src, dev_tgt),
    ])
    stats_df.to_csv(os.path.join(args.artifacts_dir,
                    "dataset_stats.csv"), index=False)

    print("[INFO] Analyzing bigram leakage...")
    leak_tbl = leakage_bigram_table(train, dev, train_src, dev_src, dev_tgt)
    leak_tbl.to_csv(os.path.join(args.artifacts_dir,
                    "leakage_bigram_overlap.csv"), index=False)

    print("[INFO] Computing Lead-k baselines (dev)...")
    baseline_dev = compute_leadk_baselines(dev, dev_src, dev_tgt, ks=(1, 2, 3))
    baseline_dev.to_csv(os.path.join(args.artifacts_dir,
                        "leadk_baseline_rouge_dev.csv"), index=False)

    print("[INFO] Computing Lead-k baselines (train)...")
    baseline_train = compute_leadk_baselines(
        train, train_src, train_tgt, ks=(1, 2, 3))
    baseline_train.to_csv(os.path.join(
        args.artifacts_dir, "leadk_baseline_rouge_train.csv"), index=False)

    bd = baseline_dev.copy()
    bd["split"] = "dev"
    bt = baseline_train.copy()
    bt["split"] = "train"
    baseline_combined = pd.concat([bt, bd], ignore_index=True)
    baseline_combined.to_csv(os.path.join(
        args.artifacts_dir, "leadk_baseline_rouge_combined.csv"), index=False)

    print("[INFO] Performing error analysis (lead-3)...")
    top_df, bottom_df = error_analysis_lead3(dev, dev_src, dev_tgt, topn=30)
    top_df.to_csv(os.path.join(args.artifacts_dir,
                  "error_analysis_lead3_top.csv"), index=False)
    bottom_df.to_csv(os.path.join(args.artifacts_dir,
                     "error_analysis_lead3_bottom.csv"), index=False)

    print("[INFO] Generating visualizations...")
    dev_len = add_len_cols(dev, dev_src, dev_tgt)
    plot_dev_distributions(dev_len, args.artifacts_dir)

    print("[INFO] Extracting top bigrams...")
    dump_top_bigrams(train, dev, train_src, dev_src,
                     dev_tgt, args.artifacts_dir, topn=100)

    print("[INFO] Computing intrasplit bigram overlap...")
    intra_train = bigram_overlap_table_pair(
        train[train_src], train[train_tgt], "train_dialogue", "train_summary")
    intra_dev = bigram_overlap_table_pair(
        dev[dev_src], dev[dev_tgt], "dev_dialogue", "dev_summary")
    intra_tbl = pd.concat([intra_train, intra_dev], ignore_index=True)
    intra_tbl.to_csv(os.path.join(args.artifacts_dir,
                     "intrasplit_bigram_overlap.csv"), index=False)

    print("[INFO] Computing lead support statistics...")
    leadpos_train = compute_lead_support_stats(train, train_src, train_tgt)
    leadpos_dev = compute_lead_support_stats(dev, dev_src, dev_tgt)

    print("[INFO] Computing reference quality metrics (morpheme-based)...")
    quality_train = compute_quality_table(train, train_src, train_tgt)
    quality_dev = compute_quality_table(dev, dev_src, dev_tgt)

    # 키워드 상세 정보 추출 (별도 CSV 파일로 저장)
    print("\n[INFO] Extracting detailed keyword information...")
    train_keyword_path = os.path.join(
        args.artifacts_dir, "keyword_details_train.csv")
    dev_keyword_path = os.path.join(
        args.artifacts_dir, "keyword_details_dev.csv")

    export_keyword_details(train, train_src, train_tgt,
                           train_keyword_path, "train")
    export_keyword_details(dev, dev_src, dev_tgt, dev_keyword_path, "dev")

    def _top_bigrams_df(series: pd.Series, n: int = 100) -> pd.DataFrame:
        cnt = Counter()
        for text in series.astype(str).values:
            t = normalize_text(text).split()
            cnt.update(ngrams(t, 2))
        items = cnt.most_common(n)
        return pd.DataFrame({"bigram": [" ".join(k) for k, _ in items], "count": [v for _, v in items]})

    top_bi_train_src = _top_bigrams_df(train[train_src], 100)
    top_bi_dev_src = _top_bigrams_df(dev[dev_src], 100)
    top_bi_dev_tgt = _top_bigrams_df(dev[dev_tgt], 100)

    print("[INFO] Creating Excel report...")
    report_path = os.path.join(args.artifacts_dir, "analysis_report.xlsx")

    with pd.ExcelWriter(report_path, engine="xlsxwriter") as xw:
        schema_overview.to_excel(xw, sheet_name="schema_overview", index=False)
        stats_df.to_excel(xw, sheet_name="dataset_stats", index=False)
        leak_tbl.to_excel(xw, sheet_name="leakage_bigram", index=False)
        intra_tbl.to_excel(xw, sheet_name="intrasplit_bigram", index=False)
        baseline_train.to_excel(
            xw, sheet_name="leadk_rouge_train", index=False)
        baseline_dev.to_excel(xw, sheet_name="leadk_rouge_dev", index=False)
        baseline_combined.to_excel(
            xw, sheet_name="leadk_rouge_combined", index=False)
        top_df.to_excel(xw, sheet_name="lead3_top", index=False)
        bottom_df.to_excel(xw, sheet_name="lead3_bottom", index=False)
        top_bi_train_src.to_excel(
            xw, sheet_name="top_bigrams_train_src", index=False)
        top_bi_dev_src.to_excel(
            xw, sheet_name="top_bigrams_dev_src", index=False)
        top_bi_dev_tgt.to_excel(
            xw, sheet_name="top_bigrams_dev_tgt", index=False)
        leadpos_train.to_excel(
            xw, sheet_name="lead_support_train", index=False)
        leadpos_dev.to_excel(xw, sheet_name="lead_support_dev", index=False)
        quality_train.to_excel(xw, sheet_name="ref_quality_train", index=False)
        quality_dev.to_excel(xw, sheet_name="ref_quality_dev", index=False)

        readme = pd.DataFrame([
            {"sheet_name": "schema_overview", "description": "원본 CSV 컬럼 스키마",
             "note": "컬럼별 평균 문자 길이 확인"},
            {"sheet_name": "dataset_stats", "description": "train/dev 기본 통계",
             "note": "행수, 결측치, 평균 길이(문자/단어/문장)"},
            {"sheet_name": "leakage_bigram", "description": "train↔dev 바이그램 겹침",
             "note": "도메인 유사도, 누설 여부"},
            {"sheet_name": "intrasplit_bigram", "description": "분할 내부 dialogue↔summary 겹침",
             "note": "추출성 지표 (낮을수록 의역 강함)"},
            {"sheet_name": "leadk_rouge_train", "description": "train Lead-k ROUGE",
             "note": "추출 베이스라인"},
            {"sheet_name": "leadk_rouge_dev", "description": "dev Lead-k ROUGE",
             "note": "모델 목표는 이 F1 상회"},
            {"sheet_name": "leadk_rouge_combined", "description": "train/dev Lead-k 비교",
             "note": "분할 간 난이도 차이"},
            {"sheet_name": "lead3_top", "description": "Lead-3 성공 사례",
             "note": "리드 바이어스 강한 패턴"},
            {"sheet_name": "lead3_bottom", "description": "Lead-3 실패 사례",
             "note": "핵심이 후반에 있는 경우"},
            {"sheet_name": "top_bigrams_train_src", "description": "train 원문 상위 바이그램",
             "note": "상투구/템플릿 확인"},
            {"sheet_name": "top_bigrams_dev_src", "description": "dev 원문 상위 바이그램",
             "note": "train과 표현 차이"},
            {"sheet_name": "top_bigrams_dev_tgt", "description": "dev 요약 상위 바이그램",
             "note": "요약 스타일 파악"},
            {"sheet_name": "lead_support_train", "description": "train 리드 지원 위치",
             "note": "앞쪽 집중이 크면 리드 바이어스"},
            {"sheet_name": "lead_support_dev", "description": "dev 리드 지원 위치",
             "note": "train과 비교해 차이 확인"},
            {"sheet_name": "ref_quality_train", "description": "train 요약 품질 (형태소 기반)",
             "note": "조사 제거 후 키워드 커버리지"},
            {"sheet_name": "ref_quality_dev", "description": "dev 요약 품질 (형태소 기반)",
             "note": "실제 키워드 겹침률 (조사 무시)"},
        ])
        readme.to_excel(xw, sheet_name="README", index=False)

    print(f"[SUCCESS] Excel report saved: {os.path.abspath(report_path)}")
    print(
        f"\n[DONE] All artifacts saved to: {os.path.abspath(args.artifacts_dir)}")
    print("\nGenerated files:")
    print("  CSV files:")
    print("    - schema_overview.csv")
    print("    - dataset_stats.csv")
    print("    - leakage_bigram_overlap.csv")
    print("    - intrasplit_bigram_overlap.csv")
    print("    - leadk_baseline_rouge_train.csv")
    print("    - leadk_baseline_rouge_dev.csv")
    print("    - leadk_baseline_rouge_combined.csv")
    print("    - error_analysis_lead3_top.csv")
    print("    - error_analysis_lead3_bottom.csv")
    print("    - top_bigrams_train_src.csv")
    print("    - top_bigrams_dev_src.csv")
    print("    - top_bigrams_dev_tgt.csv")
    print("  Keyword details (별도 CSV):")
    print("    - keyword_details_train.csv")
    print("    - keyword_details_dev.csv")
    print("  PNG files:")
    print("    - dev_length_hist.png")
    print("    - dev_src_vs_tgt_scatter.png")
    print("    - dev_compression_ratio_hist.png")
    print("    - dev_novelty_hist.png (형태소 기반)")
    print("  Excel report:")
    print("    - analysis_report.xlsx")
    print(f"\nMorpheme analyzer used: {morph_backend}")

    if morph_backend == "regex":
        print(
            "\n[WARNING] Regex fallback used - keyword analysis may be less accurate.")
        print("          Install konlpy for better morpheme analysis:")
        print("          pip install konlpy")


if __name__ == "__main__":
    main()
