"""
Screenplay-based evaluation: run a scripted user flow against generated HTML,
capture screenshots at assertion points, compute SSIM against reference.

Similar to Cloning Bench's site-test but simplified for our pipeline.

Usage:
    python screenplay_eval.py --screenplay examples/screenplay.json --html output.html
    python screenplay_eval.py --screenplay examples/screenplay.json --html output.html --ref-html reference.html

Screenplay format:
[
    {"action": "navigate", "url": "about:blank"},
    {"action": "wait", "ms": 500},
    {"action": "assert", "name": "initial_load", "screenshot": "ref_screenshots/step1.png"},
    {"action": "click", "selector": "#nav-about"},
    {"action": "wait", "ms": 300},
    {"action": "assert", "name": "after_nav_click", "screenshot": "ref_screenshots/step2.png"},
    {"action": "type", "selector": "#search", "text": "hello"},
    {"action": "assert", "name": "search_typed", "screenshot": "ref_screenshots/step3.png"},
    {"action": "hover", "selector": ".dropdown-trigger"},
    {"action": "assert", "name": "dropdown_open", "screenshot": "ref_screenshots/step4.png"}
]
"""

import argparse
import io
import json
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from playwright.sync_api import sync_playwright

from config import VIEWPORT_W, VIEWPORT_H
from reward import render_html, make_diff_image, is_full_html, _wrap_snippet


def log(msg):
    print(msg, flush=True)


def take_screenshot(page) -> np.ndarray:
    """Take a viewport screenshot as numpy array."""
    return np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


def compute_ssim(ref_img: np.ndarray, gen_img: np.ndarray) -> float:
    """Compute SSIM between two images, resizing if needed."""
    if ref_img.shape != gen_img.shape:
        gen_pil = Image.fromarray(gen_img).resize((ref_img.shape[1], ref_img.shape[0]))
        gen_img = np.array(gen_pil)
    return float(ssim_fn(ref_img, gen_img, channel_axis=2, data_range=255))


def run_screenplay(
    page,
    screenplay: list[dict],
    out_dir: str | None = None,
) -> list[dict]:
    """
    Run a screenplay against the current page content.
    Returns list of assertion results with SSIM scores.
    """
    results = []

    for i, step in enumerate(screenplay):
        action = step["action"]

        if action == "navigate":
            page.goto(step.get("url", "about:blank"))

        elif action == "wait":
            page.wait_for_timeout(step.get("ms", 300))

        elif action == "click":
            selector = step["selector"]
            try:
                page.click(selector, timeout=3000)
            except Exception as e:
                log(f"  Step {i}: click '{selector}' failed: {e}")

        elif action == "type":
            selector = step["selector"]
            text = step["text"]
            try:
                page.fill(selector, text, timeout=3000)
            except Exception as e:
                log(f"  Step {i}: type '{selector}' failed: {e}")

        elif action == "hover":
            selector = step["selector"]
            try:
                page.hover(selector, timeout=3000)
            except Exception as e:
                log(f"  Step {i}: hover '{selector}' failed: {e}")

        elif action == "assert":
            name = step.get("name", f"step_{i}")
            ref_path = step.get("screenshot")

            gen_img = take_screenshot(page)

            result = {
                "step": i,
                "name": name,
                "ssim": 0.0,
                "has_ref": ref_path is not None and os.path.exists(ref_path),
            }

            if result["has_ref"]:
                ref_img = np.array(Image.open(ref_path).convert("RGB"))
                result["ssim"] = compute_ssim(ref_img, gen_img)

                # Save outputs
                if out_dir:
                    Image.fromarray(gen_img).save(os.path.join(out_dir, f"{name}_gen.png"))
                    diff_img = make_diff_image(ref_img, gen_img, threshold=25)
                    Image.fromarray(diff_img).save(os.path.join(out_dir, f"{name}_diff.png"))

            elif out_dir:
                Image.fromarray(gen_img).save(os.path.join(out_dir, f"{name}_gen.png"))

            results.append(result)

        elif action == "scroll":
            pixels = step.get("pixels", 300)
            page.evaluate(f"window.scrollBy(0, {pixels})")
            page.wait_for_timeout(200)

        else:
            log(f"  Step {i}: unknown action '{action}'")

    return results


def generate_screenplay_from_html(
    page,
    html: str,
    out_dir: str,
) -> list[dict]:
    """
    Auto-generate a simple screenplay from HTML by:
    1. Rendering the page
    2. Taking initial screenshot
    3. Finding clickable elements
    4. Clicking each, taking screenshot after

    Returns the screenplay + saves reference screenshots.
    """
    os.makedirs(out_dir, exist_ok=True)
    render_html(page, html)
    page.wait_for_timeout(500)

    screenplay = []

    # Step 1: Initial load assertion
    ref_path = os.path.join(out_dir, "ref_initial.png")
    page.screenshot(path=ref_path)
    screenplay.append({"action": "assert", "name": "initial_load", "screenshot": ref_path})

    # Step 2: Find clickable elements
    clickables = page.evaluate("""() => {
        const elements = [];
        const els = document.querySelectorAll('a, button, [role="button"], nav a, .nav-link, [onclick]');
        for (const el of els) {
            const rect = el.getBoundingClientRect();
            if (rect.width < 5 || rect.height < 5) continue;
            if (rect.y > window.innerHeight) continue;  // skip below fold
            const text = el.innerText.trim().substring(0, 50);
            const tag = el.tagName.toLowerCase();
            // Build a selector
            let selector = '';
            if (el.id) selector = '#' + el.id;
            else if (el.className && typeof el.className === 'string')
                selector = tag + '.' + el.className.split(' ').filter(c => c).join('.');
            else selector = tag;

            elements.push({selector, text, tag, x: rect.x, y: rect.y, w: rect.width, h: rect.height});
        }
        return elements.slice(0, 10);  // max 10 clickables
    }""")

    for j, el in enumerate(clickables):
        # Click
        screenplay.append({"action": "click", "selector": el["selector"]})
        screenplay.append({"action": "wait", "ms": 300})

        # Take screenshot after click
        ref_path = os.path.join(out_dir, f"ref_click_{j}.png")
        try:
            page.click(el["selector"], timeout=2000)
            page.wait_for_timeout(300)
            page.screenshot(path=ref_path)
            screenplay.append({"action": "assert", "name": f"after_click_{j}_{el['text'][:20]}", "screenshot": ref_path})
        except Exception:
            pass

        # Reload to reset state for next click
        render_html(page, html)
        page.wait_for_timeout(300)

    # Step 3: Scroll test
    render_html(page, html)
    page.wait_for_timeout(300)
    page.evaluate("window.scrollBy(0, 400)")
    page.wait_for_timeout(300)
    ref_path = os.path.join(out_dir, "ref_scrolled.png")
    page.screenshot(path=ref_path)
    screenplay.append({"action": "scroll", "pixels": 400})
    screenplay.append({"action": "wait", "ms": 300})
    screenplay.append({"action": "assert", "name": "after_scroll", "screenshot": ref_path})

    return screenplay


def eval_html_with_screenplay(
    ref_html: str,
    gen_html: str,
    out_dir: str,
    screenplay: list[dict] | None = None,
) -> dict:
    """
    Full evaluation: generate screenplay from ref HTML (or use provided),
    run it against gen HTML, return scores.
    """
    os.makedirs(out_dir, exist_ok=True)

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})

    # Generate screenplay from reference if not provided
    if screenplay is None:
        log("Generating screenplay from reference HTML...")
        ref_dir = os.path.join(out_dir, "reference")
        screenplay = generate_screenplay_from_html(page, ref_html, ref_dir)
        with open(os.path.join(out_dir, "screenplay.json"), "w") as f:
            json.dump(screenplay, f, indent=2)
        log(f"  Generated {len([s for s in screenplay if s['action'] == 'assert'])} assertion points")

    # Run screenplay against generated HTML
    log("Running screenplay against generated HTML...")
    render_html(page, gen_html)
    page.wait_for_timeout(500)

    gen_dir = os.path.join(out_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    results = run_screenplay(page, screenplay, out_dir=gen_dir)

    browser.close()
    pw.stop()

    # Summary
    ssim_scores = [r["ssim"] for r in results if r["has_ref"]]
    summary = {
        "n_assertions": len(results),
        "avg_ssim": round(float(np.mean(ssim_scores)), 4) if ssim_scores else 0.0,
        "min_ssim": round(float(np.min(ssim_scores)), 4) if ssim_scores else 0.0,
        "max_ssim": round(float(np.max(ssim_scores)), 4) if ssim_scores else 0.0,
        "per_assertion": results,
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"\nResults:")
    for r in results:
        log(f"  {r['name']}: SSIM={r['ssim']:.3f}")
    log(f"\n  Avg SSIM: {summary['avg_ssim']:.3f}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-html", type=str, required=True, help="Reference HTML file")
    parser.add_argument("--gen-html", type=str, required=True, help="Generated HTML file")
    parser.add_argument("--screenplay", type=str, default=None, help="Screenplay JSON (auto-generated if not provided)")
    parser.add_argument("--out-dir", type=str, default="eval_output/screenplay_eval")
    args = parser.parse_args()

    with open(args.ref_html) as f:
        ref_html = f.read()
    with open(args.gen_html) as f:
        gen_html = f.read()

    screenplay = None
    if args.screenplay:
        with open(args.screenplay) as f:
            screenplay = json.load(f)

    eval_html_with_screenplay(ref_html, gen_html, args.out_dir, screenplay)


if __name__ == "__main__":
    main()
