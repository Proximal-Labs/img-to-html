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

## Base vs RL Comparison

| Config | Avg SSIM | Avg Reward | Notes |
|--------|---------|-----------|-------|
| Base 1-turn | 0.536 | -0.677 | No training |
| Base 2-turn (analyze-fix) | 0.468 | -0.658 | Analyze-fix doesn't help without training |
| **Simple RL batch 10** | **0.472** | **-0.698** | Pure SSIM reward, slight regression |

### Simple RL Batch 10 — Per-Site

| # | Website | Base SSIM | RL SSIM | Delta |
|---|---------|----------|---------|-------|
| 1 | Resy | 0.745 | 0.743 | -0.002 |
| 2 | FoxSports | 0.497 | 0.395 | -0.102 |
| 3 | UnderArmour | 0.533 | 0.429 | -0.104 |
| 4 | IKEA | 0.518 | 0.542 | +0.024 |
| 5 | Yelp | 0.471 | 0.474 | +0.003 |
| 6 | eBay | 0.716 | 0.627 | -0.089 |
| 7 | Carnival | 0.550 | 0.387 | -0.163 |
| 8 | Rentalcars | 0.396 | 0.573 | +0.177 |
| 9 | Viator | 0.322 | 0.328 | +0.006 |
| 10 | SoundCloud | 0.208 | 0.228 | +0.020 |

Simple RL (pure SSIM, no content gate) shows mixed results after 10 batches — some sites improve (Rentalcars +0.177) but others regress. Train/eval reward mismatch likely: training uses raw SSIM, eval uses content-gated SSIM.

### Visual Comparison — Resy (Base vs RL batch 10)

| Reference | Base 4B | RL Batch 10 |
|-----------|---------|-------------|
| ![ref](eval_output/m2w-4b-base-1turn/example_00/ref-render.png) | ![base](eval_output/m2w-4b-base-1turn/example_00/turn1.png) | ![rl](eval_output/m2w-simple-batch10/example_00/turn1.png) |

## Training Status

| Run | Status | Batch | Batch Time | Notes |
|-----|--------|-------|-----------|-------|
| Simple RL (1-turn) | Running | ~15/31 | ~350s | Pure SSIM reward |
| Agent RL (2-turn) | Running | ~5/31 | ~800s | Analyze-fix, content-gated reward |

Waiting for checkpoint 20 eval and agent RL results.
