# Phase 1 Attribute Test Results

**Images tested**: 20 (5 per class: MEL, NV, BCC, BKL)
**Queries**: Architecture, Network, Structures, Colors, Vessels, Regression, Keratosis, Symmetry

## Per-Image Results

| Image | GT | Lesion Attr | Arch | Net | Structures | Colors | Vessels | Regr | Kerat | Symm |
|---|---|---|---|---|---|---|---|---|---|---|
| ISIC_0024552 | MEL | atypical pigment network | irregular | yes | dots, globules | black, brown | arborizing | no | no | asymmetric |
| ISIC_0024733 | MEL | polymorphous | irregular | no | none | brown | arborizing | no | no | asymmetric |
| ISIC_0025268 | MEL | atypical pigment network | irregular | yes | dots, globules, streaks | brown | arborizing | no | no | asymmetric |
| ISIC_0025450 | MEL | shiny white lines | irregular | no | dots, globules | brown | arborizing | no | no | asymmetric |
| ISIC_0025891 | MEL | atypical streaks | oval | yes | dots, globules | black, brown | none | no | no | asymmetric |
| ISIC_0024683 | NV | fading pigment network | oval | no | dots, globules | brown | none | no | no | asymmetric |
| ISIC_0024715 | NV | homogenous | round | no | none | pink | dotted | no | no | asymmetric |
| ISIC_0024869 | NV | atypical pigment network | irregular | no | dots, globules | brown | none | no | no | asymmetric |
| ISIC_0024901 | NV | fading pigment network | round | no | dots, globules | brown | arborizing | no | no | asymmetric |
| ISIC_0024953 | NV | fading pigment network | irregular | yes | dots, globules | black, brown | arborizing | no | no | asymmetric |
| ISIC_0025019 | BCC | shiny white blotches | irregular | no | dots, globules | brown | arborizing | no | no | asymmetric |
| ISIC_0025826 | BCC | arborizing vessels | irregular | no | dots, globules | brown | arborizing | no | no | asymmetric |
| ISIC_0027189 | BCC | yellow-whitish globules | irregular | no | dots, globules | black, brown, pink | arborizing | yes | yes | asymmetric |
| ISIC_0028303 | BCC | yellow-whitish globules | irregular | no | dots, globules | brown | dotted | no | no | asymmetric |
| ISIC_0029053 | BCC | yellow-whitish globules | irregular | no | none | brown, pink | arborizing | no | no | asymmetric |
| ISIC_0024643 | BKL | gyri/ridges | oval | yes | none | brown, pink | arborizing | no | no | asymmetric |
| ISIC_0025038 | BKL | moth eaten border | oval | no | none | brown, pink | dotted | no | no | asymmetric |
| ISIC_0025529 | BKL | comedo-like openings | irregular | no | dots, globules | black, brown | arborizing | yes | no | asymmetric |
| ISIC_0025819 | BKL | clod pattern | irregular | no | dots, globules | brown | arborizing | no | no | asymmetric |
| ISIC_0026644 | BKL | comedo-like openings | oval | no | dots, globules | black, brown | dotted | yes | no | asymmetric |

## Attribute Distribution by Class

### Architecture

| Value | MEL (n=5) | NV (n=5) | BCC (n=5) | BKL (n=5) |
|---|---|---|---|---|
| irregular | **4/5 (80%)** | 2/5 (40%) | **5/5 (100%)** | 2/5 (40%) |
| oval | 1/5 (20%) | 1/5 (20%) | 0/5 (0%) | **3/5 (60%)** |
| round | 0/5 (0%) | 2/5 (40%) | 0/5 (0%) | 0/5 (0%) |

**Finding**: "irregular" dominates MEL (80%) and BCC (100%) equally — not discriminative between them. "round" is NV-only. "oval" leans BKL.

### Network

| Value | MEL (n=5) | NV (n=5) | BCC (n=5) | BKL (n=5) |
|---|---|---|---|---|
| yes | **3/5 (60%)** | 1/5 (20%) | 0/5 (0%) | 1/5 (20%) |
| no | 2/5 (40%) | **4/5 (80%)** | **5/5 (100%)** | **4/5 (80%)** |

**Finding**: Network="yes" is the strongest MEL signal here (60%) vs low in other classes. Counter-intuitive but useful. Current rules give mel_score +1 for "no" — this is **backwards**.

### Structures

| Value | MEL (n=5) | NV (n=5) | BCC (n=5) | BKL (n=5) |
|---|---|---|---|---|
| dots, globules | **3/5 (60%)** | **4/5 (80%)** | **4/5 (80%)** | **3/5 (60%)** |
| none | 1/5 (20%) | 1/5 (20%) | 1/5 (20%) | 2/5 (40%) |
| dots, globules, streaks | 1/5 (20%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |

**Finding**: "dots, globules" appears in 60-80% of ALL classes — zero discriminative power. "streaks" appeared only in MEL but only once.

### Colors — Component Analysis

| Component | MEL (n=5) | NV (n=5) | BCC (n=5) | BKL (n=5) |
|---|---|---|---|---|
| brown | **5/5 (100%)** | **4/5 (80%)** | **4/5 (80%)** | **5/5 (100%)** |
| black | 2/5 (40%) | 1/5 (20%) | 1/5 (20%) | 2/5 (40%) |
| pink | 0/5 (0%) | 1/5 (20%) | 2/5 (40%) | 2/5 (40%) |
| blue | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |

**Finding**: Colors are less discriminative than the previous color-only test suggested. Pink for BCC dropped from 60% to 40% — model is **stochastic at temperature=0.7**. Blue appeared zero times here vs 1 time previously. Black is equal MEL/BKL (40%).

**Stochasticity warning**: Comparing color-only test vs this test for same images:
- ISIC_0025019 (BCC): was "brown, pink" → now "brown" (pink disappeared)
- ISIC_0026644 (BKL): was "blue, black, brown, gray" → now "black, brown" (blue/gray disappeared)
- ISIC_0029053 (BCC): was "black, brown, pink" → now "brown, pink" (black disappeared)

### Vessels

| Value | MEL (n=5) | NV (n=5) | BCC (n=5) | BKL (n=5) |
|---|---|---|---|---|
| arborizing | **4/5 (80%)** | 2/5 (40%) | **4/5 (80%)** | **3/5 (60%)** |
| none | 1/5 (20%) | 2/5 (40%) | 0/5 (0%) | 0/5 (0%) |
| dotted | 0/5 (0%) | 1/5 (20%) | 1/5 (20%) | 2/5 (40%) |

**CRITICAL FINDING**: Model reports "arborizing" for **80% of MEL** and **60% of BKL** — not just BCC (80%). The current BCC arborizing rule catches BCC but also fires on MEL and BKL. This attribute has near-zero discriminative power for BCC.

### Regression

| Value | MEL (n=5) | NV (n=5) | BCC (n=5) | BKL (n=5) |
|---|---|---|---|---|
| no | **5/5 (100%)** | **5/5 (100%)** | **4/5 (80%)** | **3/5 (60%)** |
| yes | 0/5 (0%) | 0/5 (0%) | 1/5 (20%) | 2/5 (40%) |

**Finding**: Regression="yes" never fires for MEL (0%) despite being a textbook MEL indicator. It fires more for BKL (40%). Current mel_score gives +1 for regression — this bonus never triggers for actual MEL.

### Keratosis

| Value | MEL (n=5) | NV (n=5) | BCC (n=5) | BKL (n=5) |
|---|---|---|---|---|
| no | **5/5 (100%)** | **5/5 (100%)** | **4/5 (80%)** | **5/5 (100%)** |
| yes | 0/5 (0%) | 0/5 (0%) | 1/5 (20%) | 0/5 (0%) |

**CRITICAL FINDING**: Keratosis="yes" fires for **0% of BKL** — the BKL rule (Rule 1) is completely broken. It fired once for a BCC case instead. The model cannot detect comedo-like openings or milia-like cysts reliably.

### Symmetry

| Value | MEL (n=5) | NV (n=5) | BCC (n=5) | BKL (n=5) |
|---|---|---|---|---|
| asymmetric | **5/5 (100%)** | **5/5 (100%)** | **5/5 (100%)** | **5/5 (100%)** |
| symmetric | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |

**CRITICAL FINDING**: Model says "asymmetric" for **100% of all images** regardless of class. This attribute provides zero discriminative power. The symmetry gate on MEL rule (`if "asymmetric" in symmetry`) is always true — it's not filtering anything.

## Raw vs Cleaned — Notable Differences

- Colors prompt returns "light brown, dark brown" → collapsed to just "brown" (light/dark distinction lost)
- "red, pink" (ISIC_0024715 NV) → "pink" (red correctly filtered)
- "red, brown" (ISIC_0024901 NV) → "brown" (red correctly filtered)
- Structures raw output often includes multi-line gibberish explanations → cleaning truncates to first line
- Vessels raw output occasionally includes paragraph-length explanations → cleaning extracts keyword

## Summary: What Works vs What's Broken

### Attributes with discriminative power
| Attribute | Signal | Mechanism |
|---|---|---|
| Architecture=round | NV indicator | Only NV shows round (40%) |
| Architecture=oval | BKL indicator | BKL 60% vs others low |
| Network=yes | MEL indicator | MEL 60% vs BCC 0%, BKL 20% |
| Colors=pink | Weak BCC/BKL | 40% BCC, 40% BKL, 0% MEL |

### Attributes that are BROKEN (no discriminative power)
| Attribute | Problem |
|---|---|
| **Symmetry** | Always "asymmetric" (100% all classes) — USELESS |
| **Vessels** | "arborizing" in 60-80% of ALL classes — NOT BCC-specific |
| **Structures** | "dots, globules" in 60-80% of ALL classes — USELESS |
| **Keratosis** | Never fires for BKL (0%) — BROKEN for its intended purpose |
| **Regression** | Never fires for MEL (0%) — BROKEN for its intended purpose |

### Current rules vs reality
| Rule | Expected | Actual | Verdict |
|---|---|---|---|
| BKL: keratosis=yes | Catches BKL | 0% BKL, 20% BCC | **BROKEN** |
| BCC: arborizing vessels | Catches BCC | 80% MEL, 40% NV, 80% BCC, 60% BKL | **BROKEN** — fires everywhere |
| BCC: pink + not irregular | Catches BCC | 40% BCC, 40% BKL, 20% NV | **Weak + false positives** |
| MEL: asymmetric gate | Filters non-MEL | 100% all classes | **USELESS** — always true |
| MEL: irregular +1 | MEL signal | 80% MEL, 100% BCC, 40% NV/BKL | Weak — BCC also 100% |
| MEL: no network +1 | MEL signal | 40% MEL, 80% NV, 100% BCC | **BACKWARDS** — hurts MEL |
| MEL: regression +1 | MEL signal | 0% MEL, 40% BKL | **BROKEN** — never fires for MEL |
| MEL: blue +1 | MEL signal | 0% all classes | **DEAD CODE** |
| MEL: color_count≥3 +1 | MEL signal | Rarely any class | **DEAD CODE** |
