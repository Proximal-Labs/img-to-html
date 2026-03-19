"""
Reward function for screenshot-to-HTML RL.

Renders generated HTML to a screenshot and compares against the original
using a combination of pixel MSE and SSIM. Crops to content bounding box
to avoid the reward being dominated by matching white backgrounds.
"""

import io
import re
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from playwright.sync_api import Page

from config import VIEWPORT_W, VIEWPORT_H


def extract_html_from_response(text: str) -> str | None:
    """Extract HTML from a model response, handling thinking blocks and various formats."""
    # Try fenced code block first
    match = re.search(r"```html\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try unfenced code block
    match = re.search(r"```\s*(<!?[^`]*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If the whole response looks like HTML, use it directly
    stripped = text.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return stripped

    # Find the last HTML-like block in the response (after any thinking/reasoning)
    match = re.search(
        r"(<(?:style|div|span|h[1-6]|p|section|header|nav|main|footer|html|!DOCTYPE)[^>]*>.*)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return None


def is_full_html(html: str) -> bool:
    """Check if HTML is a full page (has doctype or <html> tag)."""
    stripped = html.strip().lower()
    return stripped.startswith("<!doctype") or stripped.startswith("<html")


def _wrap_snippet(html_snippet: str) -> str:
    """Wrap an HTML snippet in minimal boilerplate."""
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'>"
        "<style>body{margin:20px;background:#fff;}</style>"
        "</head><body>"
        f"{html_snippet}"
        "</body></html>"
    )


def render_html(page: Page, html_snippet: str):
    """Set page content from HTML snippet or full page."""
    if is_full_html(html_snippet):
        page.set_content(html_snippet)
    else:
        page.set_content(_wrap_snippet(html_snippet))
    page.wait_for_timeout(100)


def render_html_to_image(page: Page, html_snippet: str, size: int = 256) -> np.ndarray:
    """Render an HTML snippet to a numpy array."""
    render_html(page, html_snippet)
    screenshot_bytes = page.screenshot()
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB").resize((size, size))
    return np.array(img)


def render_html_to_file(page: Page, html_snippet: str | None, save_path: str) -> bool:
    """Render HTML to a screenshot file. Returns True if successful."""
    if html_snippet is None:
        Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), (240, 240, 240)).save(save_path)
        return False
    try:
        render_html(page, html_snippet)
        page.wait_for_timeout(100)
        page.screenshot(path=save_path)
        return True
    except Exception:
        Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), (240, 240, 240)).save(save_path)
        return False


def load_reference_image(path: str, size: int = 256) -> np.ndarray:
    """Load and resize a reference screenshot."""
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img)


def _get_content_bbox(img_arr: np.ndarray, bg_color: int = 255, margin: int = 5) -> tuple[int, int, int, int]:
    """Find bounding box of non-background content. Returns (y0, y1, x0, x1)."""
    diff = np.any(img_arr != bg_color, axis=2)
    rows = np.any(diff, axis=1)
    cols = np.any(diff, axis=0)

    if not rows.any():
        h, w = img_arr.shape[:2]
        return 0, h, 0, w

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    h, w = img_arr.shape[:2]
    y0 = max(0, y0 - margin)
    y1 = min(h, y1 + margin + 1)
    x0 = max(0, x0 - margin)
    x1 = min(w, x1 + margin + 1)

    return y0, y1, x0, x1


def compute_visual_reward(
    generated_html: str | None,
    reference_image: np.ndarray,
    page: Page,
    size: int = 256,
) -> float:
    """
    Compute reward for generated HTML by comparing rendered output to reference.

    Returns a float in [-1.0, 1.0]:
      - Based on pixel similarity in the content region
      - Penalizes invalid HTML or render failures
    """
    if generated_html is None:
        return -1.0

    try:
        rendered = render_html_to_image(page, generated_html, size=size)
    except Exception:
        return -1.0

    y0, y1, x0, x1 = _get_content_bbox(reference_image)
    ref_crop = reference_image[y0:y1, x0:x1]
    gen_crop = rendered[y0:y1, x0:x1]

    mse = np.mean((ref_crop.astype(float) - gen_crop.astype(float)) ** 2) / (255.0 ** 2)
    pixel_score = 1.0 - mse

    if ref_crop.shape[0] >= 7 and ref_crop.shape[1] >= 7:
        ssim_score = ssim(ref_crop, gen_crop, channel_axis=2, data_range=255)
    else:
        ssim_score = pixel_score

    # Blend: weight SSIM more for perceptual similarity
    reward = 0.2 * pixel_score + 0.8 * ssim_score

    # Scale to [-1, 1]
    reward = 2.0 * reward - 1.0

    return float(reward)
