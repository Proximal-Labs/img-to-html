# Screenshot → HTML: Research Demo

## 1. Datasets Explored

### WebSight v2 — Synthetic Websites
Synthetic but realistic pages with Tailwind CSS. Good for initial training.
- **Examples:** `data/screenshots_1024/` (1024x768 viewport)
- **Size:** 2,893 examples
- **Style:** Tailwind via CDN, inline CSS

### Design2Code — Real Websites (Simple)
484 real webpages from C4 corpus. Full HTML with Tailwind/scripts.
- **Examples:** `data/design2code/screenshots/`
- **Size:** 483 examples
- **Eval dir:** `eval_output/frontier_baselines/gpt54-hard-d2c/`

### Mind2Web — Actual Live Websites
Real screenshots of live websites: Resy, eBay, ESPN, IKEA, United Airlines, etc.
- **Examples:** `data/mind2web_landing/screenshots/`
- **Size:** 500 landing pages from 70 unique websites
- **Eval dir:** `eval_output/single_image_rl/4b-base-simple/`

| Dataset | Example Sites | Complexity | Use |
|---------|--------------|------------|-----|
| WebSight v2 | Synthetic | Low-Medium | Training |
| Design2Code | C4 real pages | Medium | Eval benchmark |
| Mind2Web | Resy, eBay, ESPN, IKEA | High | Training + Eval |

---

## 2. Tasks

### Task 1: Static Screenshot → HTML (One Shot)

Model sees one screenshot, generates HTML. Reward = SSIM.

**Base 4B on Mind2Web:**
| Site | SSIM | Ref | Gen |
|------|------|-----|-----|
| Resy | 0.816 | [ref](eval_output/single_image_rl/4b-base-simple/example_00/ref.png) | [gen](eval_output/single_image_rl/4b-base-simple/example_00/gen.png) |
| eBay | 0.727 | [ref](eval_output/single_image_rl/4b-base-simple/example_05/ref.png) | [gen](eval_output/single_image_rl/4b-base-simple/example_05/gen.png) |
| IKEA | 0.580 | [ref](eval_output/single_image_rl/4b-base-simple/example_03/ref.png) | [gen](eval_output/single_image_rl/4b-base-simple/example_03/gen.png) |
| SoundCloud | 0.206 | [ref](eval_output/single_image_rl/4b-base-simple/example_09/ref.png) | [gen](eval_output/single_image_rl/4b-base-simple/example_09/gen.png) |

**After 10 batches RL:**
| Site | Base SSIM | RL SSIM | Delta | Gen |
|------|-----------|---------|-------|-----|
| Resy | 0.816 | **0.849** | +0.033 | [gen](eval_output/single_image_rl/4b-simple-batch10/example_00/gen.png) |
| SoundCloud | 0.206 | **0.545** | +0.339 | [gen](eval_output/single_image_rl/4b-simple-batch10/example_09/gen.png) |
| eBay | 0.727 | 0.603 | -0.124 | [gen](eval_output/single_image_rl/4b-simple-batch10/example_05/gen.png) |

---

### Task 2: Multi-Turn with Analyze-Fix Agent

Model generates HTML → we render + create SSIM diff (red highlights) → model analyzes "what's wrong" → model fixes.

**Visual diff feedback:**
- [Reference](eval_output/frontier_baselines/agent_demo_v2/reference.png)
- [Turn 1 output](eval_output/frontier_baselines/agent_demo_v2/turn1.png)
- [Diff image (red = wrong)](eval_output/frontier_baselines/agent_demo_v2/diff1.png)
- [Turn 2 output](eval_output/frontier_baselines/agent_demo_v2/turn2.png)

**GPT-5.4 Analyze-Fix vs Naive (10 turns):**
```
Naive:        0.444 → peaked 0.490 → REGRESSED to 0.430
Analyze-fix:  0.442 → 0.520 → held at 0.509 (no regression)
```

**Analyze step example (GPT-5.4 actually listed the issues):**
> "heading font size is too large, paragraph line-height is too tall,
> font weight too heavy, paragraph width too narrow"

Then fixed progressively: 0.689 → 0.707 → 0.724

---

### Task 3: Interactive Flow (Action Sequences)

Model sees multiple screenshots from a user flow + action descriptions. Generates interactive HTML with JavaScript. We execute actions via Playwright and compare SSIM at each step.

**GPT-5.4 on Mind2Web flows:**
- Dir: `eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/`
- 5/5 tasks succeeded, avg initial SSIM 0.680

| Task | Site | Steps | Initial SSIM | Step SSIM | Dir |
|------|------|-------|-------------|-----------|-----|
| 1 | Budget | 6 | 0.782 | 0.709 | [task_00](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_00/) |
| 2 | SpotHero | 8 | 0.865 | 0.405 | [task_01](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_01/) |
| 3 | RottenTomatoes | 4 | 0.470 | 0.464 | [task_02](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_02/) |
| 5 | Resy | 3 | 0.817 | 0.608 | [task_04](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_04/) |

**Opus 4.6 built a full multi-page app:**
- Dir: `eval_output/flow_interactive/mind2web-opus46-3turns-textmatch/task_00/`
- Generated 31K chars with `showPage()` JavaScript routing
- Full Discogs marketplace: home → marketplace → filters → cart → checkout
- [generated.html](eval_output/flow_interactive/mind2web-opus46-3turns-textmatch/task_00/generated.html) — 21 JS event handlers

---

## 3. Long Training Runs

### Experiment 9: 27B Single-Image RL (Best Early Result)
- **Model:** Qwen3.5-27B, 90 batches
- **Data:** 1011 examples (WebSight + Design2Code)
- **Result:** Base 0.034 → RL 0.265 (+0.231), **9/10 wins**
- **Eval dir:** `eval_output/single_image_rl/exp9-batch75/`

| Example | Base | RL | Delta | Ref | Gen |
|---------|------|-----|-------|-----|-----|
| 1 | 0.061 | 0.141 | +0.079 | [ref](eval_output/single_image_rl/exp9-batch75/example_00/ref-render.png) | [rl](eval_output/single_image_rl/exp9-batch75/example_00/rl.png) |
| 7 | -0.477 | 0.272 | **+0.749** | [ref](eval_output/single_image_rl/exp9-batch75/example_06/ref-render.png) | [rl](eval_output/single_image_rl/exp9-batch75/example_06/rl.png) |
| 10 | 0.155 | 0.623 | **+0.469** | [ref](eval_output/single_image_rl/exp9-batch75/example_09/ref-render.png) | [rl](eval_output/single_image_rl/exp9-batch75/example_09/rl.png) |

### Experiment 10: 27B Overnight (200 batches)
- **Data:** 2983 examples (WebSight + D2C), resumed from exp 9
- **Result:** Base 0.429 → RL 0.596 (+0.167), 8/10 wins
- **Eval dir:** `eval_output/single_image_rl/train_20260320_045210/`

### Current Run: 4B on Mind2Web (Real Websites)
- **Model:** Qwen3.5-4B, pure SSIM reward
- **Data:** 500 Mind2Web landing pages (70 real websites)
- **Training log:** `runs/train_simple.log`
- **Eval dirs:**
  - Base: `eval_output/single_image_rl/4b-base-simple/` (avg reward 0.055)
  - Batch 5: `eval_output/single_image_rl/4b-simple-batch5/` (avg -0.041)
  - Batch 10: `eval_output/single_image_rl/4b-simple-batch10/` (avg 0.017, SoundCloud +0.339!)

---

## 4. Frontier Model Baselines

### GPT-5.4
- **Design2Code:** avg 0.716, SSIM 0.906 on hard D2C pages
  - Dir: `eval_output/frontier_baselines/gpt54-hard-d2c/`
- **Mind2Web flows:** avg initial SSIM 0.680, generates 18-22 JS handlers
  - Dir: `eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/`

### Claude Opus 4.6
- **Mind2Web:** Generates full multi-page apps with JS routing
  - Dir: `eval_output/flow_interactive/mind2web-opus46-3turns-textmatch/`

### Qwen 3.5-4B / 27B (Base)
- **Mind2Web:** SSIM ~0.53 (WebSight), ~0.05 reward on real websites
- **No JavaScript generated** — static HTML only
- **27B base:** `eval_output/single_image_rl/27b-base-simple/` (running)

---

## 5. Key Findings

### SSIM is the right reward (bitter lesson)
We tried 8-signal DOM comparison → 5 signals → 3 signals → pure SSIM. Complex rewards penalized pixel-perfect outputs. GPT-5.4 scored SSIM 0.991 but our DOM reward gave -0.5.
- **Evidence:** `eval_output/frontier_baselines/structured-gpt54-ssim-reward/`

### Analyze-then-fix prevents regression
Without analyze step, models regress after turn 5. With analyze (model lists issues before fixing), gains hold.
- **Evidence:** `eval_output/frontier_baselines/openai-gpt54-analyzefix-10turns/`

### Output is always short
Models generate 1-4K chars regardless of 100K+ char source pages. No need to filter datasets by length.
- **Evidence:** `eval_output/frontier_baselines/gpt54-hard-d2c/` — 121K char source → 4K char output, SSIM 0.85

### Frontier models generate interactive JavaScript
GPT-5.4 and Opus produce working click handlers, state management, multi-page routing from just screenshots. Qwen models generate zero JS.

---

## 6. Challenges & Solutions

| Challenge | What happened | Solution |
|-----------|--------------|----------|
| Playwright selectors | 74% of Mind2Web actions had selector=None, clicks never fired | Text-based matching: `get_by_text("Marketplace")` |
| Viewport sizes | 512x512 cut off content, 512x1536 was too tall, caused slow Tinker inference | 1024x768 (standard desktop) or 1280x720 (Mind2Web) |
| DOM reward broken | SSIM 0.991 → reward -0.5 because DOM structure differed | Pure SSIM reward |
| Reference HTML unstyled | Mind2Web `raw_html` is DOM dump without CSS | Compare against original screenshots, not re-rendered HTML |
| 27B flow training too slow | 4.5 hours per batch with multi-image prompts | One-shot single image, parallel sampling |
| HTML cutoff | 2K tokens → incomplete HTML, white screens | 16K token limit |

---

## Repo Structure

```
train_simple.py       ← Current: one-shot screenshot→HTML, pure SSIM
train_agent.py        ← Multi-turn with analyze-fix
train_agent_flow.py   ← Multi-image flow with action sequences
train_fast.py         ← All optimizations (parallel, small imgs, train-as-you-go)
eval_agent.py         ← Multi-turn eval (OpenAI, Anthropic, Tinker)
mind2web_eval.py      ← Flow eval with Playwright action sequences
multipage_eval.py     ← Multi-page generation eval
screenplay_eval.py    ← General interactive eval
reward.py             ← Pure SSIM reward
config.py             ← Shared config
EXPERIMENTS.md        ← Full experiment log
```
