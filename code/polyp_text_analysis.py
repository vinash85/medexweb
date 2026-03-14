#!/usr/bin/env python3
"""
Polyp JNET — Text Analysis of MedGemma Attribute Descriptions
==============================================================
Two-level analyses:
  Word-level (existing):
    1. TF-IDF: discriminative terms per JNET class
    2. Log-Odds with Dirichlet Prior (Monroe et al. 2008 "Fightin' Words")
    3. Keyness: Dunning's G², chi-squared, %DIFF, log ratio

  Phrase-level (new):
    Same three methods, but using extracted attribute phrases as atomic units
    instead of individual words.

Run inside Docker:
    python3 /home/project/code/polyp_text_analysis.py
"""

import os
import re
import json
import textwrap
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
from statsmodels.stats.multitest import multipletests
from wordcloud import WordCloud

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/home/project"
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "polyp_attributes")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "polyp_text_analysis")

# Also support running from host
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "polyp_attributes")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "polyp_text_analysis")

# ── JNET classes ──────────────────────────────────────────────────────────────
JNET_CLASSES = ["1", "2A", "2B", "3"]

# ── Hedge / boilerplate patterns (sentence-level removal) ────────────────────
HEDGE_PATTERNS = [
    r"might be present",
    r"possibly",
    r"could be",
    r"may or may not",
    r"difficult(ly)? (to )?(definitively )?(assess|determine|distinguish|evaluate)",
    r"cannot (be )?(definitively|reliably|clearly)",
    r"cannot confirm",
    r"cannot assess",
    r"hard to determine",
    r"not clearly (visible|delineated|visualized)",
    r"although uncertain",
    r"would require (further|additional|evaluation|review|contrast)",
    r"requires further (evaluation|assessment|review)",
    r"if clinically indicated",
    r"though not requested",
    r"please note",
    r"not diagnostic",
    r"should be performed by",
    r"based solely upon",
    r"limited by technical",
    r"this is a visual description and not",
    r"full (assessment|evaluation) would require",
    r"without (clinical context|comparison images)",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)

# Boilerplate tokens to remove after tokenization (word-level)
BOILERPLATE_TOKENS = {
    "image", "shows", "show", "showing", "appears", "appear", "appeared",
    "based", "inspection", "context", "comparison", "assessment", "evaluation",
    "finding", "findings", "view", "visible", "seen", "noted", "note",
    "provided", "depicted", "presented", "however", "although", "therefore",
    "also", "would", "could", "might", "may", "likely", "possibly",
    "suggest", "suggestive", "suggesting", "suggests",
    "one", "single", "specific", "within", "across",
    "clearly", "obviously", "obvious", "significant", "significantly",
    "approximately", "relatively", "somewhat", "slightly", "mildly",
    "overall", "general", "typically", "usually", "commonly",
    "mentioned", "described", "indicated", "consistent",
    "well", "like", "e", "g", "eg", "etc", "question",
    # Endoscopy boilerplate
    "polyp", "lesion", "observed", "visualized", "identify", "identified",
    "presence", "present", "area", "region", "feature",
}

# ── Classification leak tokens to strip ───────────────────────────────────────
CLASSIFICATION_LEAK_TOKENS = {
    "jnet", "kudo", "classification", "classified", "classify",
}

# ── Phrase extraction patterns ────────────────────────────────────────────────

# Attribute category headers to strip from phrases
ATTRIBUTE_HEADERS_RE = re.compile(
    r"^(?:"
    r"surface\s+pattern|vascular\s+pattern|color(?:\s+and)?\s*brightness|"
    r"border(?:s)?\s*(?:and\s+)?demarcation|surface\s+texture|"
    r"(?:overall\s+)?morphology|pit\s+pattern|vessel\s+(?:pattern|arrangement)|"
    r"color|brightness|demarcation|borders?"
    r")\s*[:\-–—]\s*",
    re.IGNORECASE,
)

# Leading boilerplate prefixes to strip from phrases
PHRASE_PREFIX_PATTERNS = [
    r"^the\s+(?:polyp|lesion|image|surface|border|margin|vessel|color|mucosa|mucosal)\s+"
    r"(?:shows?|appears?|has|is|demonstrates?|exhibits?|reveals?|displays?)\s+",
    r"^there\s+(?:is|are|appears?)\s+",
    r"^it\s+(?:appears?|is|has|shows?)\s+",
    r"^(?:this|these)\s+(?:shows?|appears?|suggests?|indicates?)\s+",
    r"^(?:we|i)\s+(?:can\s+)?(?:see|observe|note|identify)\s+",
    r"^(?:a|an|the)\s+",
]
PHRASE_PREFIX_RE = re.compile("|".join(PHRASE_PREFIX_PATTERNS), re.IGNORECASE)

# Bullet/list markers
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]\s*)\s*")

# ── Preprocessing ────────────────────────────────────────────────────────────

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def load_data():
    """Load all polyp_attributes JSON files, return DataFrame."""
    records = []
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.startswith("polyp_attributes_") or not fname.endswith(".json"):
            continue
        fpath = os.path.join(INPUT_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        jnet_class = data["metadata"]["jnet_class"]
        for img_name, img_data in data["images"].items():
            if img_data.get("status") != "ok":
                continue
            records.append({
                "image": img_name,
                "class": f"JNet_{jnet_class}",
                "jnet_class": jnet_class,
                "raw_text": img_data["attributes"],
            })
    df = pd.DataFrame(records)
    print(f"[DATA] Loaded {len(df)} images across {df['class'].nunique()} JNET classes")
    for cls, count in df["class"].value_counts().sort_index().items():
        print(f"  {cls}: {count}")
    return df


def filter_hedge_sentences(text):
    """Remove sentences containing hedge/speculative language."""
    sentences = sent_tokenize(text)
    kept = []
    removed = 0
    for sent in sentences:
        if HEDGE_RE.search(sent):
            removed += 1
        else:
            kept.append(sent)
    return " ".join(kept), removed


def tokenize_and_clean(text):
    """Lowercase, tokenize, remove stopwords + boilerplate + classification leaks, lemmatize."""
    tokens = word_tokenize(text.lower())
    cleaned = []
    for tok in tokens:
        if not tok.isalpha() or len(tok) < 3:
            continue
        if tok in STOPWORDS or tok in BOILERPLATE_TOKENS or tok in CLASSIFICATION_LEAK_TOKENS:
            continue
        lemma = LEMMATIZER.lemmatize(tok)
        if lemma in STOPWORDS or lemma in BOILERPLATE_TOKENS or lemma in CLASSIFICATION_LEAK_TOKENS:
            continue
        cleaned.append(lemma)
    return cleaned


# ── Phrase Extraction ────────────────────────────────────────────────────────

def normalize_phrase(phrase):
    """Normalize a phrase: lowercase, strip boilerplate prefixes, lemmatize words."""
    phrase = phrase.lower().strip()
    # Strip punctuation at edges
    phrase = phrase.strip(".,;:!?\"'()-–—")
    # Strip attribute category headers
    phrase = ATTRIBUTE_HEADERS_RE.sub("", phrase).strip()
    # Strip leading boilerplate prefixes (apply up to 2 times for nested patterns)
    for _ in range(2):
        phrase = PHRASE_PREFIX_RE.sub("", phrase).strip()
    # Strip classification leak words
    for tok in CLASSIFICATION_LEAK_TOKENS:
        phrase = re.sub(r"\b" + tok + r"\b", "", phrase, flags=re.IGNORECASE)
    # Lemmatize each word preserving order
    words = word_tokenize(phrase)
    lemmatized = []
    for w in words:
        if not w.isalpha():
            continue
        lemma = LEMMATIZER.lemmatize(w)
        lemmatized.append(lemma)
    return " ".join(lemmatized).strip()


def extract_phrases(text):
    """Extract meaningful attribute phrases from MedGemma output.

    Hierarchical segmentation:
    1. Split on newlines (catches bullet points)
    2. Strip attribute headers
    3. Split on sentence boundaries
    4. Split on commas/semicolons for clause-level fragments
    5. Normalize and filter
    """
    phrases = []

    # Split on newlines first
    lines = text.split("\n")
    chunks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip bullet markers
        line = BULLET_RE.sub("", line).strip()
        if not line:
            continue
        # Split on sentence boundaries
        sentences = sent_tokenize(line)
        for sent in sentences:
            # Split on commas and semicolons for sub-phrases
            fragments = re.split(r"[;,]\s*", sent)
            for frag in fragments:
                frag = frag.strip()
                if len(frag) < 5:
                    continue
                chunks.append(frag)

    # Normalize and filter
    for chunk in chunks:
        normed = normalize_phrase(chunk)
        # Must have at least 2 content words
        content_words = [w for w in normed.split()
                         if w not in STOPWORDS and w not in BOILERPLATE_TOKENS and len(w) >= 3]
        if len(content_words) >= 2:
            phrases.append(normed)

    return phrases


def canonicalize_phrases(all_phrase_lists):
    """Group near-duplicate phrases using Jaccard similarity on token sets.

    Returns:
        mapping: dict from original phrase -> canonical phrase
        canonical_lists: list of lists with phrases replaced by canonical forms
    """
    # Collect all unique phrases with frequencies
    phrase_freq = Counter()
    for pl in all_phrase_lists:
        phrase_freq.update(pl)

    unique_phrases = list(phrase_freq.keys())
    if len(unique_phrases) < 2:
        return {p: p for p in unique_phrases}, all_phrase_lists

    # Build token sets
    token_sets = {p: set(p.split()) for p in unique_phrases}

    # Group by Jaccard > 0.7
    mapping = {}
    used = set()
    # Sort by frequency descending so most common becomes canonical
    sorted_phrases = sorted(unique_phrases, key=lambda p: -phrase_freq[p])

    for p in sorted_phrases:
        if p in used:
            continue
        # This phrase becomes canonical for its group
        mapping[p] = p
        used.add(p)
        ts_p = token_sets[p]
        for q in sorted_phrases:
            if q in used:
                continue
            ts_q = token_sets[q]
            intersection = len(ts_p & ts_q)
            union = len(ts_p | ts_q)
            if union > 0 and intersection / union > 0.7:
                mapping[q] = p
                used.add(q)

    # Apply mapping
    canonical_lists = []
    for pl in all_phrase_lists:
        canonical_lists.append([mapping.get(p, p) for p in pl])

    n_groups = len(set(mapping.values()))
    n_orig = len(unique_phrases)
    if n_orig > n_groups:
        print(f"  [CANONICALIZE] {n_orig} unique phrases → {n_groups} canonical groups")

    return mapping, canonical_lists


def preprocess(df):
    """Apply hedge filtering + word tokenization + phrase extraction."""
    cleaned_texts = []
    token_lists = []
    phrase_lists = []
    total_hedge_removed = 0

    for _, row in df.iterrows():
        filtered, n_removed = filter_hedge_sentences(row["raw_text"])
        total_hedge_removed += n_removed
        tokens = tokenize_and_clean(filtered)
        phrases = extract_phrases(filtered)
        cleaned_texts.append(" ".join(tokens))
        token_lists.append(tokens)
        phrase_lists.append(phrases)

    df["clean_text"] = cleaned_texts
    df["tokens"] = token_lists
    df["n_tokens"] = df["tokens"].apply(len)

    # Canonicalize phrases
    _, canonical_lists = canonicalize_phrases(phrase_lists)
    df["phrases"] = canonical_lists
    df["n_phrases"] = df["phrases"].apply(len)

    total_sents = sum(len(sent_tokenize(t)) for t in df["raw_text"])
    print(f"\n[PREPROCESS] Hedge sentences removed: {total_hedge_removed}/{total_sents} "
          f"({100*total_hedge_removed/max(total_sents,1):.1f}%)")
    print(f"[PREPROCESS] Mean tokens per image after cleaning: {df['n_tokens'].mean():.0f}")
    print(f"[PREPROCESS] Mean phrases per image: {df['n_phrases'].mean():.1f}")
    total_unique_phrases = len(set(p for pl in df["phrases"] for p in pl))
    print(f"[PREPROCESS] Unique phrases (after canonicalization): {total_unique_phrases}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# WORD-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# ── TF-IDF Analysis (word) ───────────────────────────────────────────────────

def tfidf_analysis(df, top_n=25):
    """TF-IDF with unigrams + bigrams, per-class term rankings."""
    print("\n" + "=" * 60)
    print("TF-IDF ANALYSIS (word-level)")
    print("=" * 60)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
    feature_names = vectorizer.get_feature_names_out()

    classes = sorted(df["class"].unique())
    results = {}

    for cls in classes:
        mask = (df["class"] == cls).values
        class_mean = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        other_mean = np.asarray(tfidf_matrix[~mask].mean(axis=0)).flatten()
        diff = class_mean - other_mean

        top_idx = np.argsort(diff)[::-1][:top_n]
        terms = []
        for i in top_idx:
            terms.append({
                "term": feature_names[i],
                "tfidf_class": round(class_mean[i], 4),
                "tfidf_other": round(other_mean[i], 4),
                "diff": round(diff[i], 4),
            })
        results[cls] = terms
        print(f"\n  Top {top_n} discriminative terms for [{cls}]:")
        for t in terms[:10]:
            print(f"    {t['term']:30s}  diff={t['diff']:+.4f}  (class={t['tfidf_class']:.4f})")

    # Save CSV
    rows = []
    for cls, terms in results.items():
        for rank, t in enumerate(terms, 1):
            rows.append({"class": cls, "rank": rank, **t})
    csv_path = os.path.join(OUTPUT_DIR, "tfidf_top_terms.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    _plot_tfidf_heatmap(tfidf_matrix, feature_names, df, classes, top_n=15)

    return results


def _plot_tfidf_heatmap(tfidf_matrix, feature_names, df, classes, top_n=15):
    """Heatmap of top discriminative terms x classes."""
    all_top_terms = []
    seen = set()
    for cls in classes:
        mask = (df["class"] == cls).values
        class_mean = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        other_mean = np.asarray(tfidf_matrix[~mask].mean(axis=0)).flatten()
        diff = class_mean - other_mean
        for i in np.argsort(diff)[::-1]:
            term = feature_names[i]
            if term not in seen:
                all_top_terms.append(term)
                seen.add(term)
            if len(all_top_terms) >= len(classes) * top_n:
                break
        if len(all_top_terms) >= len(classes) * top_n:
            break

    all_top_terms = all_top_terms[:len(classes) * top_n]

    heat_data = np.zeros((len(all_top_terms), len(classes)))
    for j, cls in enumerate(classes):
        mask = (df["class"] == cls).values
        class_mean = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        term_to_idx = {t: i for i, t in enumerate(feature_names)}
        for i, term in enumerate(all_top_terms):
            if term in term_to_idx:
                heat_data[i, j] = class_mean[term_to_idx[term]]

    fig, ax = plt.subplots(figsize=(10, max(8, len(all_top_terms) * 0.3)))
    sns.heatmap(
        heat_data, xticklabels=classes, yticklabels=all_top_terms,
        cmap="YlOrRd", ax=ax, linewidths=0.5,
    )
    ax.set_title("TF-IDF: Top Discriminative Terms per JNET Class (word-level)")
    ax.set_xlabel("JNET Class")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "tfidf_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Log-Odds with Dirichlet Prior (word) ─────────────────────────────────────

def log_odds_dirichlet(df, top_n=25, alpha_prior=0.01):
    """Monroe, Colaresi & Quinn (2008) — Fightin' Words (word-level)."""
    print("\n" + "=" * 60)
    print("LOG-ODDS WITH DIRICHLET PRIOR (word-level)")
    print("=" * 60)

    classes = sorted(df["class"].unique())
    global_counts = Counter()
    class_counts = {}
    for cls in classes:
        tokens = []
        for tok_list in df.loc[df["class"] == cls, "tokens"]:
            tokens.extend(tok_list)
        class_counts[cls] = Counter(tokens)
        global_counts.update(tokens)

    vocab = sorted(global_counts.keys())
    vocab_size = len(vocab)
    n_total = sum(global_counts.values())

    results = {}

    for cls in classes:
        target = class_counts[cls]
        n_target = sum(target.values())

        bg = Counter()
        for other_cls in classes:
            if other_cls != cls:
                bg.update(class_counts[other_cls])
        n_bg = sum(bg.values())

        alpha_0 = alpha_prior * vocab_size

        words = []
        log_odds = []
        z_scores = []
        freqs_target = []
        freqs_bg = []

        for w in vocab:
            y_target = target[w]
            y_bg = bg[w]
            alpha_w = alpha_prior * (global_counts[w] / n_total) * vocab_size
            if alpha_w < 1e-10:
                alpha_w = alpha_prior

            log_odds_target = np.log((y_target + alpha_w) / (n_target + alpha_0 - y_target - alpha_w))
            log_odds_bg = np.log((y_bg + alpha_w) / (n_bg + alpha_0 - y_bg - alpha_w))
            delta = log_odds_target - log_odds_bg

            var = 1.0 / (y_target + alpha_w) + 1.0 / (y_bg + alpha_w)
            z = delta / np.sqrt(var)

            words.append(w)
            log_odds.append(delta)
            z_scores.append(z)
            freqs_target.append(y_target)
            freqs_bg.append(y_bg)

        lo_df = pd.DataFrame({
            "word": words,
            "log_odds": log_odds,
            "z_score": z_scores,
            "freq_target": freqs_target,
            "freq_background": freqs_bg,
            "freq_total": [freqs_target[i] + freqs_bg[i] for i in range(len(words))],
        })
        lo_df = lo_df.sort_values("z_score", ascending=False).reset_index(drop=True)
        results[cls] = lo_df

        safe = cls.lower().replace(" ", "_")
        csv_path = os.path.join(OUTPUT_DIR, f"log_odds_{safe}.csv")
        lo_df.to_csv(csv_path, index=False)

        print(f"\n  [{cls}] — top overrepresented terms (z-score):")
        for _, r in lo_df.head(10).iterrows():
            print(f"    {r['word']:30s}  z={r['z_score']:+7.2f}  "
                  f"freq={r['freq_target']}/{r['freq_background']}")
        print(f"  [{cls}] — top underrepresented terms:")
        for _, r in lo_df.tail(5).iterrows():
            print(f"    {r['word']:30s}  z={r['z_score']:+7.2f}  "
                  f"freq={r['freq_target']}/{r['freq_background']}")
        print(f"  Saved: {csv_path}")

        _plot_fightin_words(lo_df, cls, top_n)

    return results


def _plot_fightin_words(lo_df, cls, top_n=25, prefix=""):
    """Scatter: z-score vs log10(frequency), label top words."""
    fig, ax = plt.subplots(figsize=(12, 8))

    lo_df = lo_df.copy()
    lo_df["log_freq"] = np.log10(lo_df["freq_total"].clip(lower=1))

    ax.scatter(lo_df["log_freq"], lo_df["z_score"], alpha=0.15, s=8, color="grey")

    top_over = lo_df.head(top_n)
    ax.scatter(top_over["log_freq"], top_over["z_score"], alpha=0.8, s=25, color="firebrick")
    for _, r in top_over.iterrows():
        label = textwrap.shorten(r["word"], width=40, placeholder="…")
        ax.annotate(label, (r["log_freq"], r["z_score"]),
                     fontsize=7, alpha=0.9, color="firebrick",
                     ha="left", va="bottom")

    top_under = lo_df.tail(top_n)
    ax.scatter(top_under["log_freq"], top_under["z_score"], alpha=0.8, s=25, color="steelblue")
    for _, r in top_under.iterrows():
        label = textwrap.shorten(r["word"], width=40, placeholder="…")
        ax.annotate(label, (r["log_freq"], r["z_score"]),
                     fontsize=7, alpha=0.9, color="steelblue",
                     ha="left", va="top")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("log₁₀(total frequency)")
    ax.set_ylabel("z-score (log-odds ratio)")
    level = "phrase" if prefix else "word"
    ax.set_title(f"Fightin' Words ({level}): {cls} vs. rest")
    plt.tight_layout()
    safe = cls.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"{prefix}log_odds_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Keyness Analysis (word) ──────────────────────────────────────────────────

def keyness_analysis(df, top_n=25, min_freq=3):
    """Keyness metrics (word-level) for each class vs rest."""
    print("\n" + "=" * 60)
    print("KEYNESS ANALYSIS (word-level)")
    print("=" * 60)
    return _keyness_core(df, "tokens", top_n, min_freq, prefix="")


def _keyness_core(df, col, top_n=25, min_freq=3, prefix=""):
    """Shared keyness implementation for word-level and phrase-level."""
    classes = sorted(df["class"].unique())
    class_counts = {}
    for cls in classes:
        items = []
        for item_list in df.loc[df["class"] == cls, col]:
            items.extend(item_list)
        class_counts[cls] = Counter(items)

    global_counts = Counter()
    for c in class_counts.values():
        global_counts.update(c)
    vocab = sorted(global_counts.keys())

    results = {}

    for cls in classes:
        target = class_counts[cls]
        n_target = sum(target.values())

        bg = Counter()
        for other_cls in classes:
            if other_cls != cls:
                bg.update(class_counts[other_cls])
        n_bg = sum(bg.values())

        rows = []
        for w in vocab:
            a = target[w]
            b = bg[w]
            total_w = a + b

            if total_w < min_freq:
                continue

            n = n_target + n_bg

            e_a = n_target * total_w / n
            e_b = n_bg * total_w / n

            # Dunning's G²
            g2 = 0.0
            if a > 0:
                g2 += a * np.log(a / e_a)
            if b > 0:
                g2 += b * np.log(b / e_b)
            g2 *= 2
            if a / max(n_target, 1) < total_w / max(n, 1):
                g2 = -g2

            # Chi-squared
            chi2_val = 0.0
            if e_a > 0:
                chi2_val += (a - e_a) ** 2 / e_a
            if e_b > 0:
                chi2_val += (b - e_b) ** 2 / e_b

            p_val = stats.chi2.sf(abs(g2), df=1)

            freq_target_norm = (a / max(n_target, 1)) * 10000
            freq_bg_norm = (b / max(n_bg, 1)) * 10000

            if freq_bg_norm > 0:
                pct_diff = ((freq_target_norm - freq_bg_norm) / freq_bg_norm) * 100
            else:
                pct_diff = float("inf") if a > 0 else 0.0

            epsilon = 0.5
            log_ratio = np.log2((a + epsilon) / max(n_target, 1) * max(n_bg, 1) / (b + epsilon))

            rows.append({
                "word": w,
                "freq_target": a,
                "freq_background": b,
                "freq_total": total_w,
                "freq_target_norm": round(freq_target_norm, 2),
                "freq_bg_norm": round(freq_bg_norm, 2),
                "G2": round(g2, 3),
                "chi2": round(chi2_val, 3),
                "p_value": p_val,
                "pct_diff": round(pct_diff, 1),
                "log_ratio": round(log_ratio, 3),
            })

        k_df = pd.DataFrame(rows)

        if len(k_df) > 0:
            reject, pvals_corrected, _, _ = multipletests(k_df["p_value"], method="fdr_bh")
            k_df["p_adjusted"] = pvals_corrected
            k_df["significant"] = reject
        else:
            k_df["p_adjusted"] = []
            k_df["significant"] = []

        k_df = k_df.sort_values("G2", ascending=False, key=abs).reset_index(drop=True)
        results[cls] = k_df

        safe = cls.lower().replace(" ", "_")
        csv_path = os.path.join(OUTPUT_DIR, f"{prefix}keyness_{safe}.csv")
        k_df.to_csv(csv_path, index=False)

        sig_df = k_df[k_df["significant"] == True]
        print(f"\n  [{cls}] — {len(sig_df)} significant keywords (BH-corrected p<0.05)")
        print(f"  Top overrepresented (G² > 0):")
        over = k_df[k_df["G2"] > 0].head(10)
        for _, r in over.iterrows():
            star = "*" if r["significant"] else " "
            w_display = textwrap.shorten(r["word"], width=35, placeholder="…")
            print(f"   {star} {w_display:35s}  G²={r['G2']:+8.2f}  LR={r['log_ratio']:+6.2f}  "
                  f"%DIFF={r['pct_diff']:+8.1f}  freq={r['freq_target']}/{r['freq_background']}")
        print(f"  Top underrepresented (G² < 0):")
        under = k_df[k_df["G2"] < 0].head(10)
        for _, r in under.iterrows():
            star = "*" if r["significant"] else " "
            w_display = textwrap.shorten(r["word"], width=35, placeholder="…")
            print(f"   {star} {w_display:35s}  G²={r['G2']:+8.2f}  LR={r['log_ratio']:+6.2f}  "
                  f"%DIFF={r['pct_diff']:+8.1f}  freq={r['freq_target']}/{r['freq_background']}")
        print(f"  Saved: {csv_path}")

        _plot_volcano(k_df, cls, prefix=prefix)

    return results


def _plot_volcano(k_df, cls, prefix=""):
    """Volcano plot: log_ratio (effect size) vs -log10(p_adjusted)."""
    if len(k_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    k_df = k_df.copy()
    k_df["neg_log_p"] = -np.log10(k_df["p_adjusted"].clip(lower=1e-300))

    ns = k_df[~k_df["significant"]]
    ax.scatter(ns["log_ratio"], ns["neg_log_p"], alpha=0.3, s=10, color="grey", label="NS")

    sig = k_df[k_df["significant"]]
    sig_over = sig[sig["log_ratio"] > 0]
    sig_under = sig[sig["log_ratio"] < 0]

    ax.scatter(sig_over["log_ratio"], sig_over["neg_log_p"],
               alpha=0.7, s=20, color="firebrick", label="Overrepresented")
    ax.scatter(sig_under["log_ratio"], sig_under["neg_log_p"],
               alpha=0.7, s=20, color="steelblue", label="Underrepresented")

    for subset, color, ha, va in [
        (sig_over.head(15), "firebrick", "left", "bottom"),
        (sig_under.head(15), "steelblue", "left", "top"),
    ]:
        for _, r in subset.iterrows():
            label = textwrap.shorten(r["word"], width=40, placeholder="…")
            ax.annotate(label, (r["log_ratio"], r["neg_log_p"]),
                         fontsize=6, alpha=0.85, color=color, ha=ha, va=va)

    ax.axhline(-np.log10(0.05), color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Log₂ Ratio (effect size)")
    ax.set_ylabel("-log₁₀(adjusted p-value)")
    level = "phrase" if prefix else "word"
    ax.set_title(f"Keyness Volcano ({level}): {cls} vs. rest")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    safe = cls.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"{prefix}keyness_volcano_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Word Clouds (word-level) ─────────────────────────────────────────────────

def generate_wordclouds(df):
    """Word cloud per JNET class from cleaned token frequencies."""
    print("\n" + "=" * 60)
    print("WORD CLOUDS")
    print("=" * 60)

    classes = sorted(df["class"].unique())
    for cls in classes:
        tokens = []
        for tok_list in df.loc[df["class"] == cls, "tokens"]:
            tokens.extend(tok_list)
        freq = Counter(tokens)

        if not freq:
            continue

        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            max_words=100,
            colormap="viridis",
        ).generate_from_frequencies(freq)

        safe = cls.lower().replace(" ", "_")
        path = os.path.join(OUTPUT_DIR, f"wordcloud_{safe}.png")
        wc.to_file(path)
        print(f"  Saved: {path}")


# ── Summary Report (word-level) ──────────────────────────────────────────────

def summary_report(tfidf_results, logodds_results, keyness_results):
    """Cross-method summary: terms flagged by 2+ methods."""
    print("\n" + "=" * 60)
    print("CROSS-METHOD CONSENSUS (word-level)")
    print("=" * 60)

    lines = [
        "=" * 70,
        "Polyp JNET Text Analysis — Cross-Method Consensus Report (word-level)",
        "=" * 70,
        "",
    ]

    classes = sorted(tfidf_results.keys())

    for cls in classes:
        tfidf_terms = set(t["term"] for t in tfidf_results[cls][:30] if " " not in t["term"])
        logodds_top = logodds_results[cls].head(30)
        logodds_terms = set(logodds_top["word"])
        keyness_top = keyness_results[cls]
        keyness_sig = keyness_top[(keyness_top["G2"] > 0) & (keyness_top["significant"] == True)]
        keyness_terms = set(keyness_sig.head(30)["word"])

        all_terms = tfidf_terms | logodds_terms | keyness_terms
        consensus = []
        for term in all_terms:
            count = sum([
                term in tfidf_terms,
                term in logodds_terms,
                term in keyness_terms,
            ])
            if count >= 2:
                methods = []
                if term in tfidf_terms:
                    methods.append("TF-IDF")
                if term in logodds_terms:
                    methods.append("LogOdds")
                if term in keyness_terms:
                    methods.append("Keyness")
                consensus.append((term, count, ", ".join(methods)))

        consensus.sort(key=lambda x: -x[1])

        lines.append(f"\n{'─' * 60}")
        lines.append(f"  {cls}")
        lines.append(f"{'─' * 60}")
        lines.append(f"  Consensus terms (in 2+ methods): {len(consensus)}")
        for term, count, methods in consensus:
            lines.append(f"    {term:30s}  [{count}/3]  ({methods})")

        print(f"\n  [{cls}] — {len(consensus)} consensus terms")
        for term, count, methods in consensus[:8]:
            print(f"    {term:30s}  [{count}/3]  ({methods})")

    report_text = "\n".join(lines) + "\n"
    path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(path, "w") as f:
        f.write(report_text)
    print(f"\n  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PHRASE-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# ── TF-IDF Analysis (phrase) ─────────────────────────────────────────────────

def phrase_tfidf_analysis(df, top_n=25):
    """TF-IDF with pre-extracted phrases as atomic terms."""
    print("\n" + "=" * 60)
    print("TF-IDF ANALYSIS (phrase-level)")
    print("=" * 60)

    vectorizer = TfidfVectorizer(
        analyzer=lambda doc: doc,  # input is pre-tokenized phrase lists
        max_features=3000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(df["phrases"])
    feature_names = vectorizer.get_feature_names_out()

    classes = sorted(df["class"].unique())
    results = {}

    for cls in classes:
        mask = (df["class"] == cls).values
        class_mean = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        other_mean = np.asarray(tfidf_matrix[~mask].mean(axis=0)).flatten()
        diff = class_mean - other_mean

        top_idx = np.argsort(diff)[::-1][:top_n]
        terms = []
        for i in top_idx:
            terms.append({
                "term": feature_names[i],
                "tfidf_class": round(class_mean[i], 4),
                "tfidf_other": round(other_mean[i], 4),
                "diff": round(diff[i], 4),
            })
        results[cls] = terms
        print(f"\n  Top discriminative phrases for [{cls}]:")
        for t in terms[:10]:
            p_display = textwrap.shorten(t["term"], width=45, placeholder="…")
            print(f"    {p_display:45s}  diff={t['diff']:+.4f}  (class={t['tfidf_class']:.4f})")

    # Save CSV
    rows = []
    for cls, terms in results.items():
        for rank, t in enumerate(terms, 1):
            rows.append({"class": cls, "rank": rank, **t})
    csv_path = os.path.join(OUTPUT_DIR, "phrase_tfidf_top_terms.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    _plot_phrase_tfidf_heatmap(tfidf_matrix, feature_names, df, classes, top_n=10)

    return results


def _plot_phrase_tfidf_heatmap(tfidf_matrix, feature_names, df, classes, top_n=10):
    """Heatmap of top discriminative phrases x classes."""
    all_top = []
    seen = set()
    for cls in classes:
        mask = (df["class"] == cls).values
        class_mean = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        other_mean = np.asarray(tfidf_matrix[~mask].mean(axis=0)).flatten()
        diff = class_mean - other_mean
        for i in np.argsort(diff)[::-1]:
            term = feature_names[i]
            if term not in seen:
                all_top.append(term)
                seen.add(term)
            if len(all_top) >= len(classes) * top_n:
                break
        if len(all_top) >= len(classes) * top_n:
            break

    all_top = all_top[:len(classes) * top_n]

    heat_data = np.zeros((len(all_top), len(classes)))
    for j, cls in enumerate(classes):
        mask = (df["class"] == cls).values
        class_mean = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        term_to_idx = {t: i for i, t in enumerate(feature_names)}
        for i, term in enumerate(all_top):
            if term in term_to_idx:
                heat_data[i, j] = class_mean[term_to_idx[term]]

    # Truncate long phrase labels
    y_labels = [textwrap.shorten(t, width=50, placeholder="…") for t in all_top]

    fig, ax = plt.subplots(figsize=(12, max(8, len(all_top) * 0.4)))
    sns.heatmap(
        heat_data, xticklabels=classes, yticklabels=y_labels,
        cmap="YlOrRd", ax=ax, linewidths=0.5,
    )
    ax.set_title("TF-IDF: Top Discriminative Phrases per JNET Class")
    ax.set_xlabel("JNET Class")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "phrase_tfidf_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Log-Odds with Dirichlet Prior (phrase) ───────────────────────────────────

def phrase_log_odds_dirichlet(df, top_n=25, alpha_prior=0.01):
    """Fightin' Words at the phrase level."""
    print("\n" + "=" * 60)
    print("LOG-ODDS WITH DIRICHLET PRIOR (phrase-level)")
    print("=" * 60)

    classes = sorted(df["class"].unique())
    global_counts = Counter()
    class_counts = {}
    for cls in classes:
        phrases = []
        for phrase_list in df.loc[df["class"] == cls, "phrases"]:
            phrases.extend(phrase_list)
        class_counts[cls] = Counter(phrases)
        global_counts.update(phrases)

    vocab = sorted(global_counts.keys())
    vocab_size = len(vocab)
    n_total = sum(global_counts.values())

    if vocab_size == 0:
        print("  No phrases found. Skipping.")
        return {}

    results = {}

    for cls in classes:
        target = class_counts[cls]
        n_target = sum(target.values())

        bg = Counter()
        for other_cls in classes:
            if other_cls != cls:
                bg.update(class_counts[other_cls])
        n_bg = sum(bg.values())

        alpha_0 = alpha_prior * vocab_size

        words = []
        log_odds_vals = []
        z_scores = []
        freqs_target = []
        freqs_bg = []

        for w in vocab:
            y_target = target[w]
            y_bg = bg[w]
            alpha_w = alpha_prior * (global_counts[w] / n_total) * vocab_size
            if alpha_w < 1e-10:
                alpha_w = alpha_prior

            lo_target = np.log((y_target + alpha_w) / (n_target + alpha_0 - y_target - alpha_w))
            lo_bg = np.log((y_bg + alpha_w) / (n_bg + alpha_0 - y_bg - alpha_w))
            delta = lo_target - lo_bg

            var = 1.0 / (y_target + alpha_w) + 1.0 / (y_bg + alpha_w)
            z = delta / np.sqrt(var)

            words.append(w)
            log_odds_vals.append(delta)
            z_scores.append(z)
            freqs_target.append(y_target)
            freqs_bg.append(y_bg)

        lo_df = pd.DataFrame({
            "word": words,
            "log_odds": log_odds_vals,
            "z_score": z_scores,
            "freq_target": freqs_target,
            "freq_background": freqs_bg,
            "freq_total": [freqs_target[i] + freqs_bg[i] for i in range(len(words))],
        })
        lo_df = lo_df.sort_values("z_score", ascending=False).reset_index(drop=True)
        results[cls] = lo_df

        safe = cls.lower().replace(" ", "_")
        csv_path = os.path.join(OUTPUT_DIR, f"phrase_log_odds_{safe}.csv")
        lo_df.to_csv(csv_path, index=False)

        print(f"\n  [{cls}] — top overrepresented phrases (z-score):")
        for _, r in lo_df.head(10).iterrows():
            p_display = textwrap.shorten(r["word"], width=45, placeholder="…")
            print(f"    {p_display:45s}  z={r['z_score']:+7.2f}  "
                  f"freq={r['freq_target']}/{r['freq_background']}")
        print(f"  [{cls}] — top underrepresented phrases:")
        for _, r in lo_df.tail(5).iterrows():
            p_display = textwrap.shorten(r["word"], width=45, placeholder="…")
            print(f"    {p_display:45s}  z={r['z_score']:+7.2f}  "
                  f"freq={r['freq_target']}/{r['freq_background']}")
        print(f"  Saved: {csv_path}")

        _plot_fightin_words(lo_df, cls, top_n, prefix="phrase_")

    return results


# ── Keyness Analysis (phrase) ─────────────────────────────────────────────────

def phrase_keyness_analysis(df, top_n=25, min_freq=2):
    """Keyness metrics (phrase-level) for each class vs rest."""
    print("\n" + "=" * 60)
    print("KEYNESS ANALYSIS (phrase-level)")
    print("=" * 60)
    return _keyness_core(df, "phrases", top_n, min_freq, prefix="phrase_")


# ── Phrase Bar Charts ────────────────────────────────────────────────────────

def generate_phrase_barcharts(df, top_n=20):
    """Horizontal bar chart of top phrases per JNET class."""
    print("\n" + "=" * 60)
    print("PHRASE BAR CHARTS")
    print("=" * 60)

    classes = sorted(df["class"].unique())
    for cls in classes:
        phrases = []
        for phrase_list in df.loc[df["class"] == cls, "phrases"]:
            phrases.extend(phrase_list)
        freq = Counter(phrases)

        if not freq:
            continue

        top = freq.most_common(top_n)
        labels = [textwrap.shorten(p, width=50, placeholder="…") for p, _ in top]
        values = [c for _, c in top]

        fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.35)))
        y_pos = range(len(labels))
        ax.barh(y_pos, values, color="steelblue", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top {top_n} Attribute Phrases: {cls}")
        plt.tight_layout()
        safe = cls.lower().replace(" ", "_")
        path = os.path.join(OUTPUT_DIR, f"phrase_barchart_{safe}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Phrase Summary Report ────────────────────────────────────────────────────

def phrase_summary_report(tfidf_results, logodds_results, keyness_results):
    """Cross-method summary for phrase-level: phrases flagged by 2+ methods."""
    print("\n" + "=" * 60)
    print("CROSS-METHOD CONSENSUS (phrase-level)")
    print("=" * 60)

    lines = [
        "=" * 70,
        "Polyp JNET Text Analysis — Cross-Method Consensus Report (phrase-level)",
        "=" * 70,
        "",
    ]

    classes = sorted(tfidf_results.keys())

    for cls in classes:
        tfidf_terms = set(t["term"] for t in tfidf_results[cls][:30])
        logodds_top = logodds_results[cls].head(30)
        logodds_terms = set(logodds_top["word"])
        keyness_top = keyness_results[cls]
        keyness_sig = keyness_top[(keyness_top["G2"] > 0) & (keyness_top["significant"] == True)]
        keyness_terms = set(keyness_sig.head(30)["word"])

        all_terms = tfidf_terms | logodds_terms | keyness_terms
        consensus = []
        for term in all_terms:
            count = sum([
                term in tfidf_terms,
                term in logodds_terms,
                term in keyness_terms,
            ])
            if count >= 2:
                methods = []
                if term in tfidf_terms:
                    methods.append("TF-IDF")
                if term in logodds_terms:
                    methods.append("LogOdds")
                if term in keyness_terms:
                    methods.append("Keyness")
                consensus.append((term, count, ", ".join(methods)))

        consensus.sort(key=lambda x: -x[1])

        lines.append(f"\n{'─' * 60}")
        lines.append(f"  {cls}")
        lines.append(f"{'─' * 60}")
        lines.append(f"  Consensus phrases (in 2+ methods): {len(consensus)}")
        for term, count, methods in consensus:
            lines.append(f"    {term:50s}  [{count}/3]  ({methods})")

        print(f"\n  [{cls}] — {len(consensus)} consensus phrases")
        for term, count, methods in consensus[:8]:
            p_display = textwrap.shorten(term, width=45, placeholder="…")
            print(f"    {p_display:45s}  [{count}/3]  ({methods})")

    report_text = "\n".join(lines) + "\n"
    path = os.path.join(OUTPUT_DIR, "phrase_summary_report.txt")
    with open(path, "w") as f:
        f.write(report_text)
    print(f"\n  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    if len(df) == 0:
        print("No data found. Run polyp_attributes.py first to generate attribute descriptions.")
        return

    df = preprocess(df)

    # Save preprocessed data
    preproc_path = os.path.join(OUTPUT_DIR, "preprocessed.csv")
    df[["image", "class", "clean_text", "n_tokens", "n_phrases"]].to_csv(preproc_path, index=False)
    print(f"  Saved preprocessed data: {preproc_path}")

    # ── Word-level analyses ──────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  WORD-LEVEL ANALYSES".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    tfidf_results = tfidf_analysis(df)
    logodds_results = log_odds_dirichlet(df)
    keyness_results = keyness_analysis(df)
    generate_wordclouds(df)
    summary_report(tfidf_results, logodds_results, keyness_results)

    # ── Phrase-level analyses ────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  PHRASE-LEVEL ANALYSES".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    p_tfidf_results = phrase_tfidf_analysis(df)
    p_logodds_results = phrase_log_odds_dirichlet(df)
    p_keyness_results = phrase_keyness_analysis(df)
    generate_phrase_barcharts(df)
    if p_tfidf_results and p_logodds_results and p_keyness_results:
        phrase_summary_report(p_tfidf_results, p_logodds_results, p_keyness_results)

    print("\n" + "=" * 60)
    print("DONE — all results in: " + OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
