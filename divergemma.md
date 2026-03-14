# divergemma — Branch Plan & Progress

## Goal
Analyze MedGemma's free-text attribute descriptions across **three medical imaging domains** using corpus linguistics methods to find **discriminative language patterns** per class.

### Domains
1. **CT Lung Cancer** — 5 classes (adenocarcinoma, large cell carcinoma, squamous cell carcinoma, benign, normal)
2. **Polyp JNET** — 4 JNET classes (1, 2A, 2B, 3) from NBI endoscopic images
3. **Skin Lesion** — 6 Dx classes (nv, mel, bkl, bcc, akiec, df) from dermoscopic images

---

## Pipeline (same for each domain)

### Step 1: Attribute Extraction (`*_attributes.py`)
- Load images, provide ground-truth diagnosis to MedGemma
- MedGemma describes visual attributes supporting the diagnosis
- Resume support (skips already-processed images)
- Output: one JSON per class in `data/<domain>_attributes/`

### Step 2: Text Analysis (`*_text_analysis.py`)
Two-level analyses:

**Word-level:**
1. **TF-IDF** — discriminative terms per class (unigrams + bigrams)
2. **Log-Odds with Dirichlet Prior** (Monroe et al. 2008 "Fightin' Words") — regularized log-odds with z-scores
3. **Keyness** — Dunning's G², chi-squared, %DIFF, log ratio

**Phrase-level:**
Same three methods, but using extracted attribute phrases as atomic units instead of individual words.

### Preprocessing (shared across analyses)
- Hedge filtering: strip speculative/filler sentences
- Tokenize, lowercase, remove stopwords, lemmatize (NLTK)
- Remove domain-specific boilerplate tokens
- BH-corrected p-values, minimum frequency thresholds

### Visualizations
- TF-IDF heatmaps, per-class bar charts
- Fightin' Words plots (z-score vs frequency)
- Volcano plots (effect size vs significance)
- Word clouds per class

---

## Scripts

| Script | Domain | Purpose |
|--------|--------|---------|
| `code/ct_lung_attributes.py` | CT Lung | Attribute extraction (5 cancer type subfolders) |
| `code/ct_lung_text_analysis.py` | CT Lung | Word + phrase-level text analysis |
| `code/polyp_attributes.py` | Polyp | Attribute extraction (JNET class from filename) |
| `code/polyp_text_analysis.py` | Polyp | Word + phrase-level text analysis |
| `code/skin_lesion_attributes.py` | Skin | Attribute extraction (Dx from DermsGemms.csv) |
| `code/skin_lesion_text_analysis.py` | Skin | Word + phrase-level text analysis |

---

## Data Layout

### CT Lung
- **Images**: `data/ct_lung_images/images/<subfolder>/` (adenocarcinoma, large cell carcinoma, squamous cell carcinoma, Benign cases, Normal cases)
- **Attributes**: `data/ct_lung_attributes/ct_lung_attributes_<class>.json`
- **Analysis output**: `data/ct_lung_text_analysis/`

### Polyp JNET
- **Images**: `data/polyp/` (flat folder, JNET class parsed from filename e.g. `23001_1_Tubular_LGD_JNet_2A.jpg`)
- **Attributes**: `data/polyp_attributes/polyp_attributes_jnet_<class>.json`
- **Analysis output**: `data/polyp_text_analysis/`

### Skin Lesion
- **Images**: `data/images/` (ISIC_*.jpg)
- **Annotations**: `code/DermsGemms.csv` (Image, Dx, Lesion attributes)
- **Attributes**: `data/skin_lesion_attributes/skin_lesion_attributes_<dx>.json`
- **Analysis output**: `data/skin_lesion_text_analysis/`

---

## Running (in Docker)

### Install dependencies
```bash
pip install wordcloud
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Attribute extraction
```bash
python3 /home/project/code/ct_lung_attributes.py --limit 50
python3 /home/project/code/polyp_attributes.py --limit 100
python3 /home/project/code/skin_lesion_attributes.py          # default: all images
```

### Text analysis (run after extraction completes)
```bash
python3 /home/project/code/ct_lung_text_analysis.py
python3 /home/project/code/polyp_text_analysis.py
python3 /home/project/code/skin_lesion_text_analysis.py
```

---

## Libraries

### Already installed
| Package | Version | Use |
|---------|---------|-----|
| scikit-learn | 1.4.2 | TF-IDF vectorization |
| scipy | 1.13.1 | Chi-squared, log-likelihood stats |
| pandas | 2.2.2 | DataFrames, CSV I/O |
| numpy | 1.26.4 | Array ops, log-odds math |
| matplotlib | 3.8.4 | Plotting |
| seaborn | 0.13.2 | Heatmaps, styled plots |
| nltk | 3.8.1 | Tokenization, stopwords, lemmatization |
| statsmodels | 0.14.2 | Multiple testing correction (BH) |

### Need to install
| Package | Use |
|---------|-----|
| wordcloud | Word cloud visualizations per class |

---

## Task Tracker

| # | Task | Status |
|---|------|--------|
| 1 | CT lung attribute extraction (`--limit 50`) | **done** |
| 2 | CT lung text analysis (word + phrase level) | **done** |
| 3 | Polyp attribute extraction script | **done** |
| 4 | Polyp text analysis script | **done** |
| 5 | Skin lesion attribute extraction script | **done** |
| 6 | Skin lesion text analysis script | **done** |
| 7 | Run polyp extraction + analysis | pending |
| 8 | Run skin lesion extraction + analysis | pending |
| 9 | Cross-domain comparison of results | pending |

---

## Notes
- All scripts follow the same architecture: singleton model load, KV cache reset, resume support, contrast-enhanced 224x224 input
- Branch: `divergemma` (from `vlm-trial`)
- `data/` directory is gitignored (output data not committed)
