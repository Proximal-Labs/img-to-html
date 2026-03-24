# prox — Screenshot-to-HTML RL Training

RL-train open-source VLMs (Qwen 3.5-4B/27B) to reproduce real websites from screenshots using [Tinker](https://docs.tinker.dev/).

## What it does

Model sees a screenshot of a real website → generates HTML/CSS/JS that reproduces it. We train with GRPO + PPO using SSIM-based reward, with multi-turn analyze-fix for self-correction.

## Results

**4B model on Mind2Web real websites (Resy, eBay, IKEA, etc.):**

| | Avg SSIM | Avg Reward |
|---|---------|-----------|
| Base (no RL) | 0.536 | -0.677 |
| RL (10 batches, single-shot) | TBD | TBD |
| RL (2-turn analyze-fix) | TBD | TBD |

See [DEMO.md](DEMO.md) for visual comparisons and [EXPERIMENTS.md](EXPERIMENTS.md) for full experiment log.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install tinker-cookbook
playwright install chromium
export TINKER_API_KEY=your-key-here
```

## Training

```bash
# Single-shot: screenshot → HTML (fastest, ~350s/batch)
MODEL=Qwen/Qwen3.5-4B MANIFEST_PATH=data/mind2web_landing/manifest.json \
  RENDERER_NAME=qwen3_disable_thinking \
  python train_simple.py

# Multi-turn agent: generate → analyze → fix (2 turns, ~800s/batch)
MODEL=Qwen/Qwen3.5-4B MANIFEST_PATH=data/mind2web_landing/manifest.json \
  RENDERER_NAME=qwen3_disable_thinking MAX_TURNS=2 \
  python train_agent.py

# Interactive flow: action sequences with Playwright (~1900s/batch)
MODEL=Qwen/Qwen3.5-4B RENDERER_NAME=qwen3_disable_thinking \
  MAX_ACTION_STEPS=3 \
  python train_flow.py
```

## Eval

```bash
# Base model eval (parallel — fires all prompts at once)
python eval_agent.py --n 10 --turns 2 --provider tinker --name my-eval

# RL model eval
python eval_agent.py --n 10 --turns 2 --provider tinker \
  --model_path "tinker://session-id/sampler_weights/name" --name my-rl-eval

# Flow eval (Mind2Web action sequences)
python mind2web_eval.py --n 5 --turns 2 --provider tinker
```

## Key Config (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3.5-27B` | Base model |
| `BATCH_SIZE` | `16` | Prompts per batch |
| `GROUP_SIZE` | `4` | Rollouts per prompt |
| `MAX_TURNS` | `2` | Agent turns (analyze-fix) |
| `TOKENS_PER_TURN` | `8192` | Max generation tokens |
| `KL_BETA` | `0.02` | KL penalty |
| `SAVE_EVERY` | `5` | Checkpoint frequency |
| `MANIFEST_PATH` | `data/manifest.json` | Dataset path |

## Reward

Content-gated SSIM + text + color:

```
content_gate = 0.2 + 0.8 * max(text_match, color_match)
reward = 2 * (0.60 * ssim * content_gate + 0.25 * text + 0.15 * color) - 1
```

Blank pages score -0.78. Perfect match scores +1.0. No hard gates.

## Files

```
config.py           # Shared configuration
reward.py           # Reward function (content-gated SSIM)
train_simple.py     # Single-shot RL training
train_agent.py      # Multi-turn agent RL (analyze-fix)
train_flow.py       # Interactive flow RL (action sequences)
eval_agent.py       # Eval with parallel inference
mind2web_eval.py    # Flow eval on Mind2Web
DEMO.md             # Visual results and comparisons
EXPERIMENTS.md      # Full experiment log (16 experiments)
EVAL.md             # Latest eval report
```
