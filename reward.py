"""
Reward function for screenshot-to-HTML RL.

Content-gated SSIM + text + color signals.
SSIM is gated by content presence to prevent blank page reward hacking.

Reward = 2 * (0.60 * gated_ssim + 0.25 * text + 0.15 * color) - 1
"""

import io
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from playwright.sync_api import Page

from config import VIEWPORT_W, VIEWPORT_H


# ── HTML extraction ───────────────────────────────────────────────────────────

def extract_html_from_response(text: str) -> str | None:
    """Extract HTML from a model response."""
    match = re.search(r"```html\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(<!?[^`]*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    stripped = text.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return stripped

    match = re.search(
        r"(<(?:style|div|span|h[1-6]|p|section|header|nav|main|footer|html|!DOCTYPE)[^>]*>.*)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return None


# ── HTML rendering ────────────────────────────────────────────────────────────

def is_full_html(html: str) -> bool:
    stripped = html.strip().lower()
    return stripped.startswith("<!doctype") or stripped.startswith("<html")


def _wrap_snippet(html_snippet: str) -> str:
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'>"
        "<style>body{margin:20px;background:#fff;}</style>"
        "</head><body>"
        f"{html_snippet}"
        "</body></html>"
    )


def render_html(page: Page, html_snippet: str):
    try:
        if is_full_html(html_snippet):
            page.set_content(html_snippet, timeout=5000)
        else:
            page.set_content(_wrap_snippet(html_snippet), timeout=5000)
    except Exception:
        # If set_content times out, try with a blank page first
        page.set_content("<html><body></body></html>", timeout=2000)
        return
    try:
        page.wait_for_load_state("networkidle", timeout=3000)
    except Exception:
        page.wait_for_timeout(200)


def render_html_to_image(page: Page, html_snippet: str, size: int = 256) -> np.ndarray:
    render_html(page, html_snippet)
    screenshot_bytes = page.screenshot()
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB").resize((size, size))
    return np.array(img)


def render_html_to_file(page: Page, html_snippet: str | None, save_path: str, full_page: bool = True) -> bool:
    if html_snippet is None:
        Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), (240, 240, 240)).save(save_path)
        return False
    try:
        render_html(page, html_snippet)
        page.wait_for_timeout(100)
        page.screenshot(path=save_path, full_page=full_page)
        return True
    except Exception:
        Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), (240, 240, 240)).save(save_path)
        return False


# ── Visual diff for agent feedback ─────────────────────────────────────────────

def make_diff_image(ref_img: np.ndarray, gen_img: np.ndarray, threshold: int = 25) -> np.ndarray:
    """
    Create a diff image highlighting pixel differences in red.
    Regions where ref and gen differ beyond threshold are overlaid in red.
    Returns an RGB numpy array.
    """
    # Ensure same size
    if ref_img.shape != gen_img.shape:
        gen_pil = Image.fromarray(gen_img).resize((ref_img.shape[1], ref_img.shape[0]))
        gen_img = np.array(gen_pil)

    diff = np.abs(ref_img.astype(int) - gen_img.astype(int))
    mask = np.any(diff > threshold, axis=2)

    # Blend: gen image with red overlay on differing pixels
    result = gen_img.copy()
    result[mask] = (
        (result[mask].astype(int) * 0.3 + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
    )
    return result


# ── Combined DOM extraction (single JS call) ─────────────────────────────────

MEANINGFUL_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "a", "button", "input",
    "img", "nav", "header", "footer", "main", "section", "article",
    "li", "td", "th", "label", "span", "textarea", "select",
}


def extract_dom_info(page: Page) -> dict:
    """
    Extract all DOM info in a single JS evaluate call.
    Returns {text, blocks, colors} — no redundant round-trips.
    """
    tags_list = list(MEANINGFUL_TAGS)
    return page.evaluate("""(tagsList) => {
        const meaningfulTags = new Set(tagsList);
        const blocks = [];
        const colors = [];
        const els = document.querySelectorAll('*');

        for (const el of els) {
            const rect = el.getBoundingClientRect();
            if (rect.width < 5 || rect.height < 5) continue;

            const tag = el.tagName.toLowerCase();
            const style = getComputedStyle(el);

            // Skip invisible elements
            if (style.display === 'none' || style.visibility === 'hidden' ||
                parseFloat(style.opacity) === 0) continue;

            // Colors (from all visible elements)
            if (rect.width >= 10 && rect.height >= 10) {
                const bg = style.backgroundColor;
                if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                    const m = bg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                    if (m) colors.push([parseInt(m[1]), parseInt(m[2]), parseInt(m[3])]);
                }
                const fg = style.color;
                if (fg) {
                    const m = fg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                    if (m) colors.push([parseInt(m[1]), parseInt(m[2]), parseInt(m[3])]);
                }
            }

            // Blocks (meaningful elements only)
            if (rect.width >= window.innerWidth * 0.99 && rect.height >= window.innerHeight * 0.99) continue;

            const hasBg = style.backgroundColor !== 'rgba(0, 0, 0, 0)' && style.backgroundColor !== 'transparent';
            const hasBorder = style.borderWidth && style.borderWidth !== '0px';

            let directText = '';
            for (const node of el.childNodes) {
                if (node.nodeType === Node.TEXT_NODE) {
                    directText += node.textContent;
                }
            }
            directText = directText.trim();

            if (meaningfulTags.has(tag) || directText.length > 0 || hasBg || hasBorder) {
                blocks.push({
                    tag: tag,
                    x: rect.x,
                    y: rect.y,
                    w: rect.width,
                    h: rect.height,
                    text: directText.substring(0, 200),
                    fontSize: parseFloat(style.fontSize) || 0,
                    fontWeight: style.fontWeight || '400',
                    color: style.color || '',
                    bgColor: style.backgroundColor || '',
                    borderRadius: style.borderRadius || '0px',
                    padding: style.padding || '0px',
                });
            }
        }

        return {
            text: document.body ? document.body.innerText.trim() : '',
            blocks: blocks,
            colors: colors,
        };
    }""", tags_list)


def extract_ref_info(page: Page, reference_html: str, size: int = 256) -> dict:
    """
    Extract all reference info in one shot. Call once per prompt, reuse
    across all rollouts in the group.

    Returns {text, blocks, colors, image}.
    """
    render_html(page, reference_html)
    dom = extract_dom_info(page)
    screenshot_bytes = page.screenshot()
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB").resize((size, size))
    dom["image"] = np.array(img)
    return dom


def extract_gen_info(page: Page, generated_html: str, size: int = 256) -> dict:
    """
    Render generated HTML once, extract DOM + screenshot in same page load.
    """
    render_html(page, generated_html)
    dom = extract_dom_info(page)
    screenshot_bytes = page.screenshot()
    full_img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    dom["image_full"] = np.array(full_img)  # Full res for blank check
    dom["image"] = np.array(full_img.resize((size, size)))  # 256x256 for SSIM
    return dom


# ── Comparison functions ─────────────────────────────────────────────────────

def text_similarity(ref_text: str, gen_text: str) -> float:
    """Global text content similarity."""
    if not ref_text and not gen_text:
        return 1.0
    if not ref_text or not gen_text:
        return 0.0
    return SequenceMatcher(None, ref_text.lower(), gen_text.lower()).ratio()


def _quantize_color(c, step: int = 32):
    return (c[0] // step * step, c[1] // step * step, c[2] // step * step)


def color_palette_similarity(ref_colors: list, gen_colors: list) -> float:
    if not ref_colors and not gen_colors:
        return 1.0
    if not ref_colors or not gen_colors:
        return 0.0

    ref_hist = Counter(_quantize_color(tuple(c)) for c in ref_colors)
    gen_hist = Counter(_quantize_color(tuple(c)) for c in gen_colors)

    all_colors = set(ref_hist.keys()) | set(gen_hist.keys())
    intersection = sum(min(ref_hist.get(c, 0), gen_hist.get(c, 0)) for c in all_colors)
    total = max(sum(ref_hist.values()), sum(gen_hist.values()))

    return float(intersection / total) if total > 0 else 0.0


def visual_similarity(ref_img: np.ndarray, gen_img: np.ndarray) -> float:
    """Straight SSIM on full images, no cropping."""
    if ref_img.shape != gen_img.shape:
        from PIL import Image
        gen_img = np.array(Image.fromarray(gen_img).resize((ref_img.shape[1], ref_img.shape[0])))
    return float(ssim(ref_img, gen_img, channel_axis=2, data_range=255))


# ── Combined reward ──────────────────────────────────────────────────────────

def compute_reward_from_info(ref_info: dict, gen_info: dict) -> tuple[float, dict]:
    """
    Compute reward: SSIM gated by content presence, plus text and color signals.

    SSIM is multiplied by a content gate so blank pages can't ride high SSIM
    against light backgrounds. The gate opens smoothly as text/color appear.
    No hard -1 penalties — the gradient is continuous everywhere.
    """
    details = {
        "ssim": visual_similarity(ref_info["image"], gen_info["image"]),
        "text": text_similarity(ref_info["text"], gen_info["text"]),
        "color": color_palette_similarity(ref_info["colors"], gen_info["colors"]),
    }

    # Content gate: SSIM only gets full credit when the page has content
    # Blank page (text=0, color=0) → gate=0.2, SSIM contribution crushed
    # Real attempt (text>0 or color>0) → gate opens toward 1.0
    content = max(details["text"], details["color"])
    content_gate = 0.2 + 0.8 * content
    gated_ssim = details["ssim"] * content_gate

    raw = 0.60 * gated_ssim + 0.25 * details["text"] + 0.15 * details["color"]

    # Scale to [-1, 1]
    reward = 2.0 * raw - 1.0

    details["content_gate"] = content_gate
    return float(reward), details


