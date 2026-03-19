"""
Download and prepare website HTML/screenshot pairs from HuggingFaceM4/WebSight.

Uses WebSight v0.1 which has synthetic but realistic pages with inline CSS —
no external dependencies, renders fully offline via Playwright.

Filters for HTML that:
  - Is short enough to fit in ~1024 tokens (≤ MAX_HTML_CHARS)
  - Renders meaningful visual content at 512×512 (not blank/broken)

Outputs the same manifest.json format as generate_dataset.py.
"""

import json
import io
import os
import re
import random

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
SCREENSHOTS_DIR = os.path.join(OUT_DIR, "screenshots")
VIEWPORT_W, VIEWPORT_H = 512, 512

# Character budget — roughly 500–600 tokens for Qwen tokenizer
MAX_HTML_CHARS = 2000

# Minimum fraction of non-white pixels to accept a render
MIN_CONTENT_RATIO = 0.02

# How many samples to aim for
WEBSIGHT_TARGET = 256
WEBSIGHT_CANDIDATES = 3000  # download this many to filter from


# ── Utilities ─────────────────────────────────────────────────────────────────

def is_full_html(html: str) -> bool:
    """Check if HTML is a full page (has doctype or <html> tag)."""
    stripped = html.strip().lower()
    return stripped.startswith("<!doctype") or stripped.startswith("<html")


def extract_body_content(html: str) -> str | None:
    """Extract body innerHTML from a full HTML page."""
    match = re.search(r"<body[^>]*>(.*)</body>", html, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_inline_styles(html: str) -> str:
    """Extract <style> blocks from a full HTML page."""
    styles = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
    if styles:
        return "<style>" + "\n".join(styles) + "</style>"
    return ""


def has_meaningful_content(img_arr: np.ndarray, min_ratio: float = MIN_CONTENT_RATIO) -> bool:
    """Check if rendered screenshot has enough non-white pixels."""
    diff = np.any(img_arr != 255, axis=2)
    return diff.sum() / diff.size > min_ratio


def render_html(page, html: str) -> np.ndarray | None:
    """Render HTML in Playwright and return image array, or None on failure."""
    try:
        if is_full_html(html):
            page.set_content(html)
        else:
            wrapped = (
                "<!DOCTYPE html>"
                "<html><head><meta charset='utf-8'>"
                "<style>body{margin:20px;background:#fff;}</style>"
                "</head><body>"
                f"{html}"
                "</body></html>"
            )
            page.set_content(wrapped)

        page.wait_for_timeout(300)
        screenshot_bytes = page.screenshot()
        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        return np.array(img)
    except Exception:
        return None


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_websight_samples(n_target: int = WEBSIGHT_TARGET,
                           n_candidates: int = WEBSIGHT_CANDIDATES) -> list[dict]:
    """Download and filter samples from WebSight v0.1 (inline CSS, no external deps)."""
    from datasets import load_dataset

    print(f"Loading WebSight v0.1 ({n_candidates} candidates)...")
    ds = load_dataset(
        "HuggingFaceM4/WebSight",
        name="v0.1",
        split=f"train[:{n_candidates}]",
        trust_remote_code=True,
    )

    samples = []
    skipped_long = 0

    for i, row in enumerate(ds):
        html = row["text"]

        if len(html) > MAX_HTML_CHARS:
            skipped_long += 1
            continue

        # For full pages, extract body + style as the snippet
        # (what the model will need to generate)
        if is_full_html(html):
            body = extract_body_content(html)
            styles = extract_inline_styles(html)
            if body is None:
                continue
            snippet = (styles + "\n" + body).strip() if styles else body
            if len(snippet) > MAX_HTML_CHARS:
                skipped_long += 1
                continue
        else:
            snippet = html

        samples.append({
            "html": snippet,
            "full_html": html,  # keep original for rendering
            "source": "websight",
            "source_idx": i,
        })

    print(f"  Passed char filter: {len(samples)} "
          f"(skipped {skipped_long} too long)")

    if len(samples) > n_target:
        samples = random.sample(samples, n_target)
        print(f"  Sampled down to {n_target}")

    return samples


# ── Rendering pipeline ────────────────────────────────────────────────────────

def render_and_save(samples: list[dict]) -> list[dict]:
    """Render all samples to screenshots, filtering out broken/blank ones."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    manifest = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})

        for i, sample in enumerate(samples):
            # Render using full HTML if available, otherwise wrap the snippet
            render_source = sample.get("full_html", sample["html"])
            img_arr = render_html(page, render_source)

            if img_arr is None:
                continue

            if not has_meaningful_content(img_arr):
                continue

            idx = len(manifest)
            screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{idx:04d}.png")
            Image.fromarray(img_arr).save(screenshot_path)

            manifest.append({
                "id": idx,
                "screenshot": screenshot_path,
                "html": sample["html"],  # the snippet the model should generate
            })

            if len(manifest) % 50 == 0:
                print(f"  Rendered {len(manifest)} valid samples...")

        browser.close()

    return manifest


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(42)

    # Fetch from WebSight
    samples = fetch_websight_samples()
    random.shuffle(samples)
    print(f"\nTotal candidates after filtering: {len(samples)}")

    # Render and do final visual check
    print("\nRendering and validating screenshots...")
    manifest = render_and_save(samples)

    # Save manifest
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Saved {len(manifest)} pairs to {manifest_path}")

    # Print stats on HTML lengths
    lengths = [len(m["html"]) for m in manifest]
    if lengths:
        print(f"\nHTML length stats:")
        print(f"  min: {min(lengths)} chars")
        print(f"  max: {max(lengths)} chars")
        print(f"  mean: {sum(lengths) / len(lengths):.0f} chars")
        print(f"  median: {sorted(lengths)[len(lengths)//2]} chars")


if __name__ == "__main__":
    main()
