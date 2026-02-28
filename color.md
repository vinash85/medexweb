# Color Attribute Test Results

Prompt used:
> "What colors are present in this skin lesion? Check for: light brown, dark brown, black, blue-gray, white, pink. List only colors clearly visible, separated by commas."

## Per-Image Results

| Image | GT | Lesion Attributes | Colors Detected | Count | Black | Blue | Pink | Red (raw) |
|-------|----|--------------------|-----------------|-------|-------|------|------|-----------|
| ISIC_0024552 | MEL | atypical pigment network | black, brown | 2 | Y | - | - | - |
| ISIC_0024733 | MEL | polymorphous | brown | 1 | - | - | - | - |
| ISIC_0025268 | MEL | atypical pigment network | brown | 1 | - | - | - | - |
| ISIC_0025450 | MEL | shiny white lines | black, brown | 2 | Y | - | - | - |
| ISIC_0025891 | MEL | atypical streaks | black, brown | 2 | Y | - | - | - |
| ISIC_0024683 | NV | fading pigment network | brown | 1 | - | - | - | - |
| ISIC_0024715 | NV | homogenous | pink | 1 | - | - | Y | - |
| ISIC_0024869 | NV | atypical pigment network | brown | 1 | - | - | - | - |
| ISIC_0024901 | NV | fading pigment network | brown | 1 | - | - | - | Y |
| ISIC_0024953 | NV | fading pigment network | black, brown | 2 | Y | - | - | Y |
| ISIC_0025019 | BCC | shiny white blotches and strands | brown, pink | 2 | - | - | Y | - |
| ISIC_0025826 | BCC | arborizing/branched vessels | brown | 1 | - | - | - | - |
| ISIC_0027189 | BCC | multiple-aggregated yellow-whitish globules | brown, pink | 2 | - | - | Y | - |
| ISIC_0028303 | BCC | multiple-aggregated yellow-whitish globules | brown | 1 | - | - | - | - |
| ISIC_0029053 | BCC | multiple-aggregated yellow-whitish globules | black, brown, pink | 3 | Y | - | Y | - |
| ISIC_0024643 | BKL | gyri/ridges | brown, pink | 2 | - | - | Y | - |
| ISIC_0025038 | BKL | moth eaten border | brown, pink | 2 | - | - | Y | - |
| ISIC_0025529 | BKL | comedo-like openings | black, brown | 2 | Y | - | - | - |
| ISIC_0025819 | BKL | clod pattern | brown | 1 | - | - | - | - |
| ISIC_0026644 | BKL | comedo-like openings | blue, black, brown, gray | 4 | Y | Y | - | - |

## Summary by Class

| Signal | MEL (n=5) | NV (n=5) | BCC (n=5) | BKL (n=5) |
|--------|-----------|----------|-----------|-----------|
| Avg color count | 1.6 | 1.2 | 1.8 | 2.2 |
| Has black | **60%** | 20% | 20% | 40% |
| Has blue | 0% | 0% | 0% | 20% |
| Has pink | 0% | 20% | **60%** | 40% |
| Red in raw output | 0% | 40% | 0% | 0% |

## Key Findings

### What the data supports
1. **Pink → BCC**: 60% of BCC have pink. Strongest color signal for BCC.
2. **Black → MEL**: 60% of MEL have black. Strongest color signal for MEL.
3. **Single brown → NV**: Most NV are brown-only (60%), consistent with benign nevi.

### What the data does NOT support
1. **Blue → MEL**: 0% of MEL have blue. Blue only appeared in 1 BKL case (ISIC_0026644).
2. **Color count ≥ 3 → MEL**: MEL avg is 1.6. Only BKL reached 3+ colors. This rule would hurt BKL, not help MEL.

### Problems with current rules
1. **Pink BCC rule has false positives**: 20% NV (ISIC_0024715) and 40% BKL also show pink. The `irregular not in arch` guard does not prevent NV misclassification since NV is typically round/oval.
2. **Blue MEL bonus is dead code**: Zero MEL cases have blue in this sample.
3. **Color count ≥ 3 MEL bonus is counterproductive**: Only BKL hits this threshold.

### Red leakage
- Model still returns "red" in 40% of NV raw output despite prompt not asking for it.
- `clean_response` correctly filters it out (not in `known_colors` list).
- No red leakage in cleaned output.

## Recommendations
- Replace blue + color_count MEL bonuses with **black** as MEL indicator
- Pink BCC rule needs stronger guard to avoid stealing NV/BKL cases
- Consider requiring pink + another BCC signal (e.g., vessels not none, or symmetric shape)
