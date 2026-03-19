"""
Eval: compare base Qwen3.5-4B vs RL-trained model on held-out screenshots.

Saves per-example:
  - reference screenshot (copied from dataset)
  - base model generated HTML + rendered screenshot
  - RL model generated HTML + rendered screenshot
  - reward scores

Usage:
    python eval.py                     # auto-loads RL model from last train
    python eval.py --n 20              # more examples
    python eval.py --model_path "tinker://..."  # explicit model path
    python eval.py --base_only         # base-only eval
"""

import argparse
import io
import json
import os
import random
import shutil

import numpy as np
import tinker
from tinker import types
from PIL import Image
from playwright.sync_api import sync_playwright

from tinker_cookbook import renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoImageProcessor
from reward import (
    extract_html_from_response,
    load_reference_image,
    compute_visual_reward,
    _is_full_html,
)

MODEL = "Qwen/Qwen3.5-4B"
LORA_RANK = 32
MAX_TOKENS = 1024
IMG_SIZE = 256
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "data", "manifest.json")
MODEL_PATH_FILE = "/tmp/prox-rl/model_path.txt"

SYSTEM_PROMPT = (
    "You are an expert at converting screenshots of web pages into HTML/CSS code. "
    "Given a screenshot, output ONLY the HTML/CSS code that reproduces the visual appearance. "
    "Use inline styles or a <style> block. Do not include <html>, <head>, or <body> wrapper tags. "
    "Wrap your code in ```html ... ```."
)


def log(msg):
    print(msg, flush=True)


def build_prompt(renderer, screenshot_path: str):
    img = Image.open(screenshot_path).convert("RGB")
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Generate the HTML/CSS snippet that reproduces this screenshot."},
            ],
        },
    ]
    return renderer.build_generation_prompt(convo)


def render_html_to_screenshot(page, html_snippet: str, save_path: str):
    """Render HTML to a screenshot file. Returns True if successful."""
    if html_snippet is None:
        # Save a blank/error image
        img = Image.new("RGB", (512, 512), (240, 240, 240))
        img.save(save_path)
        return False

    try:
        if _is_full_html(html_snippet):
            page.set_content(html_snippet)
        else:
            full_html = (
                "<!DOCTYPE html>"
                "<html><head><meta charset='utf-8'>"
                "<style>body{margin:20px;background:#fff;}</style>"
                "</head><body>"
                f"{html_snippet}"
                "</body></html>"
            )
            page.set_content(full_html)
        page.wait_for_timeout(200)
        page.screenshot(path=save_path)
        return True
    except Exception:
        img = Image.new("RGB", (512, 512), (240, 240, 240))
        img.save(save_path)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of eval examples")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Tinker model path for RL weights")
    parser.add_argument("--base_only", action="store_true", help="Only eval base model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Find RL model path
    rl_model_path = args.model_path
    if not rl_model_path and not args.base_only:
        if os.path.exists(MODEL_PATH_FILE):
            with open(MODEL_PATH_FILE) as f:
                rl_model_path = f.read().strip()
            log(f"Loaded RL model path: {rl_model_path}")
        else:
            log(f"No model path found at {MODEL_PATH_FILE}. Run train.py first, or pass --model_path.")
            log("Running base-only eval.")
            args.base_only = True

    with open(MANIFEST_PATH) as f:
        dataset = json.load(f)

    samples = random.sample(dataset, min(args.n, len(dataset)))

    # Setup
    tokenizer = get_tokenizer(MODEL)
    image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    renderer = renderers.get_renderer("qwen3_5_disable_thinking", tokenizer, image_processor=image_processor)

    eval_params = types.SamplingParams(
        max_tokens=MAX_TOKENS, stop=renderer.get_stop_sequences(), temperature=0.3,
    )

    service_client = tinker.ServiceClient()

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    reward_pages = [browser.new_page(viewport={"width": 512, "height": 512}) for _ in range(8)]
    render_page = browser.new_page(viewport={"width": 512, "height": 512})

    out_dir = os.path.join(os.path.dirname(__file__), "eval_output")
    os.makedirs(out_dir, exist_ok=True)

    # Base model
    log("\nCreating base model (untrained LoRA)...")
    base_client = service_client.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)
    base_sampler = base_client.save_weights_and_get_sampling_client()

    # RL model
    rl_sampler = None
    if not args.base_only:
        log(f"Loading RL model...")
        rl_sampler = service_client.create_sampling_client(model_path=rl_model_path)

    # Evaluate
    base_rewards, rl_rewards = [], []

    for i, item in enumerate(samples):
        example_dir = os.path.join(out_dir, f"example_{i:02d}")
        os.makedirs(example_dir, exist_ok=True)

        screenshot_path = item["screenshot"]
        ref_img = load_reference_image(screenshot_path, size=IMG_SIZE)
        page = reward_pages[i % len(reward_pages)]
        prompt = build_prompt(renderer, screenshot_path)

        # Copy reference screenshot
        shutil.copy2(screenshot_path, os.path.join(example_dir, "reference.png"))

        # ── Base model ────────────────────────────────────────────────────
        base_result = base_sampler.sample(prompt=prompt, num_samples=1, sampling_params=eval_params).result()
        base_parsed, _ = renderer.parse_response(base_result.sequences[0].tokens)
        base_html = extract_html_from_response(get_text_content(base_parsed))
        base_reward = compute_visual_reward(base_html, ref_img, page)
        base_rewards.append(base_reward)

        # Save base HTML and render screenshot
        with open(os.path.join(example_dir, "base.html"), "w") as f:
            f.write(base_html or "<!-- No HTML generated -->")
        render_html_to_screenshot(render_page, base_html, os.path.join(example_dir, "base.png"))

        # ── RL model ──────────────────────────────────────────────────────
        rl_reward = None
        if rl_sampler is not None:
            rl_result = rl_sampler.sample(prompt=prompt, num_samples=1, sampling_params=eval_params).result()
            rl_parsed, _ = renderer.parse_response(rl_result.sequences[0].tokens)
            rl_html = extract_html_from_response(get_text_content(rl_parsed))
            rl_reward = compute_visual_reward(rl_html, ref_img, page)
            rl_rewards.append(rl_reward)

            with open(os.path.join(example_dir, "rl.html"), "w") as f:
                f.write(rl_html or "<!-- No HTML generated -->")
            render_html_to_screenshot(render_page, rl_html, os.path.join(example_dir, "rl.png"))

        # Save metadata
        meta = {
            "screenshot": screenshot_path,
            "ground_truth_html": item["html"],
            "base_reward": round(base_reward, 4),
        }
        if rl_reward is not None:
            meta["rl_reward"] = round(rl_reward, 4)
            meta["delta"] = round(rl_reward - base_reward, 4)
        with open(os.path.join(example_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Log
        if rl_reward is not None:
            delta = rl_reward - base_reward
            marker = ">" if delta > 0 else ("<" if delta < 0 else "=")
            log(f"  {i+1}/{len(samples)}: base={base_reward:.3f} {marker} rl={rl_reward:.3f}  "
                f"(delta={delta:+.3f})  {os.path.basename(screenshot_path)}")
        else:
            log(f"  {i+1}/{len(samples)}: base={base_reward:.3f}  {os.path.basename(screenshot_path)}")

    # ── Summary ───────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("EVAL RESULTS")
    log(f"{'='*60}")

    avg_base = float(np.mean(base_rewards))
    log(f"  Base model:  mean={avg_base:.3f}  std={np.std(base_rewards):.3f}  "
        f"min={np.min(base_rewards):.3f}  max={np.max(base_rewards):.3f}")

    summary = {
        "base_mean": round(avg_base, 4),
        "n_eval": len(samples),
    }

    if rl_rewards:
        avg_rl = float(np.mean(rl_rewards))
        wins = sum(1 for b, r in zip(base_rewards, rl_rewards) if r > b)

        log(f"  RL model:    mean={avg_rl:.3f}  std={np.std(rl_rewards):.3f}  "
            f"min={np.min(rl_rewards):.3f}  max={np.max(rl_rewards):.3f}")
        log(f"  Improvement: {avg_rl - avg_base:+.3f}")
        log(f"  RL wins: {wins}/{len(samples)}")

        summary.update({
            "rl_mean": round(avg_rl, 4),
            "improvement": round(avg_rl - avg_base, 4),
            "rl_wins": wins,
            "rl_model_path": rl_model_path,
        })

    with open(os.path.join(out_dir, "eval_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nOutputs saved to {out_dir}/")
    log(f"Each example_XX/ folder has: reference.png, base.png, base.html, rl.png, rl.html, meta.json")

    browser.close()
    pw.stop()


if __name__ == "__main__":
    main()
