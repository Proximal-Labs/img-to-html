# Eval Report: Base 4B on Mind2Web (1-turn, single shot)

**Date:** 2026-03-23
**Model:** Qwen 3.5-4B (base, no RL)
**Dataset:** Mind2Web landing pages (500 real websites), seed=42
**Turns:** 1 (single shot, no analyze-fix)
**Reference:** Original website screenshots (not DOM re-renders)
**Reward:** Content-gated SSIM (0.60) + text (0.25) + color (0.15)

## Summary

| Metric | Value |
|--------|-------|
| Avg Reward | -0.677 |
| Avg SSIM | 0.536 |
| Best SSIM | 0.745 (Resy) |
| Worst SSIM | 0.208 (SoundCloud) |

## Per-Example Results

| # | Website | SSIM | Reward | Notes |
|---|---------|------|--------|-------|
| 1 | Resy | 0.745 | -0.789 | Best SSIM — got layout + branding colors |
| 2 | FoxSports | 0.497 | -0.906 | Dark theme missed |
| 3 | UnderArmour | 0.533 | -0.902 | Complex hero banner |
| 4 | IKEA | 0.518 | -0.600 | Got text content, missed styling |
| 5 | Yelp | 0.471 | -0.150 | Best reward — decent match |
| 6 | eBay | 0.716 | -0.757 | Good layout, high SSIM |
| 7 | Carnival | 0.550 | -0.861 | Complex cruise site |
| 8 | Rentalcars | 0.396 | -0.576 | Form-heavy page |
| 9 | Viator | 0.322 | -0.730 | Complex travel layout |
| 10 | SoundCloud | 0.208 | -0.496 | Dark theme, completely missed |

## Visual Comparisons

### Resy (SSIM 0.745) — Best result
| Reference | Base 4B |
|-----------|---------|
| ![ref](eval_output/m2w-4b-base-1turn/example_00/ref-render.png) | ![gen](eval_output/m2w-4b-base-1turn/example_00/turn1.png) |

### eBay (SSIM 0.716) — Good layout match
| Reference | Base 4B |
|-----------|---------|
| ![ref](eval_output/m2w-4b-base-1turn/example_05/ref-render.png) | ![gen](eval_output/m2w-4b-base-1turn/example_05/turn1.png) |

### IKEA (SSIM 0.518) — Got text, missed styling
| Reference | Base 4B |
|-----------|---------|
| ![ref](eval_output/m2w-4b-base-1turn/example_03/ref-render.png) | ![gen](eval_output/m2w-4b-base-1turn/example_03/turn1.png) |

### SoundCloud (SSIM 0.208) — Worst result, dark theme missed
| Reference | Base 4B |
|-----------|---------|
| ![ref](eval_output/m2w-4b-base-1turn/example_09/ref-render.png) | ![gen](eval_output/m2w-4b-base-1turn/example_09/turn1.png) |

## Comparison: 1-turn vs 2-turn (analyze-fix)

| Config | Avg SSIM | Avg Reward |
|--------|---------|-----------|
| 1-turn (single shot) | 0.536 | -0.677 |
| 2-turn (analyze-fix) | 0.468 | -0.658 |

Analyze-fix barely helps on base model (+0.019 reward). The model isn't trained to use visual feedback. RL training should improve this.

## Training Status

| Run | Status | Batch Time | Notes |
|-----|--------|-----------|-------|
| Simple RL (1-turn) | Running | ~350s/batch | Pure SSIM reward, batch 2 done |
| Agent RL (2-turn) | Running | ~1244s/batch | Analyze-fix, batch 0 done |

Will eval simple RL at checkpoints 10 and 20 to measure improvement.
