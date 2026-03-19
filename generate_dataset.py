"""
Generate synthetic (screenshot, HTML/CSS) pairs for RL training.

Produces simple HTML snippets, renders each to a PNG screenshot via Playwright,
and saves a manifest JSON mapping screenshot paths to ground-truth HTML.
"""

import json
import os
import random
import itertools
from pathlib import Path
from playwright.sync_api import sync_playwright


COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e", "#ecf0f1", "#d35400"]
FONT_SIZES = ["14px", "18px", "24px", "32px", "48px"]
PADDINGS = ["5px", "10px", "15px", "20px", "30px"]
BORDER_RADII = ["0px", "4px", "8px", "12px", "50%"]
WIDTHS = ["60px", "100px", "150px", "200px", "300px"]
HEIGHTS = ["40px", "60px", "80px", "100px", "150px"]
TEXTS = ["Hello", "Click Me", "Submit", "Welcome", "Learn More", "Sign Up", "OK", "Cancel", "Next", "Done"]
FONT_FAMILIES = ["Arial", "Georgia", "Courier New", "Verdana", "Times New Roman"]
BORDER_STYLES = ["none", "1px solid black", "2px solid #333", "1px dashed #999", "2px dotted #666"]


def generate_snippets():
    """Generate a list of simple HTML snippets with variety."""
    snippets = []

    # --- Colored boxes ---
    for color, w, h, radius in itertools.islice(
        itertools.product(COLORS, WIDTHS, HEIGHTS, BORDER_RADII), 40
    ):
        snippets.append(
            f'<div style="background:{color};width:{w};height:{h};border-radius:{radius};"></div>'
        )

    # --- Headings ---
    for text, color, size, font in itertools.islice(
        itertools.product(TEXTS, COLORS, FONT_SIZES, FONT_FAMILIES), 40
    ):
        snippets.append(
            f'<h1 style="color:{color};font-size:{size};font-family:{font};">{text}</h1>'
        )

    # --- Buttons ---
    for text, bg, color, pad, radius in itertools.islice(
        itertools.product(TEXTS, COLORS, COLORS, PADDINGS, BORDER_RADII), 40
    ):
        if bg == color:
            continue
        snippets.append(
            f'<button style="background:{bg};color:{color};padding:{pad};border-radius:{radius};border:none;font-size:16px;cursor:pointer;">{text}</button>'
        )

    # --- Paragraphs ---
    para_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
    ]
    for text, color, size, font in itertools.islice(
        itertools.product(para_texts, COLORS, FONT_SIZES[:3], FONT_FAMILIES), 30
    ):
        snippets.append(
            f'<p style="color:{color};font-size:{size};font-family:{font};">{text}</p>'
        )

    # --- Simple two-element layouts ---
    for _ in range(30):
        bg1, bg2 = random.sample(COLORS, 2)
        w = random.choice(WIDTHS)
        h = random.choice(HEIGHTS)
        gap = random.choice(["5px", "10px", "20px"])
        snippets.append(
            f'<div style="display:flex;gap:{gap};">'
            f'<div style="background:{bg1};width:{w};height:{h};"></div>'
            f'<div style="background:{bg2};width:{w};height:{h};"></div>'
            f'</div>'
        )

    # --- Input fields ---
    for pad, radius, border, font in itertools.islice(
        itertools.product(PADDINGS, BORDER_RADII[:3], BORDER_STYLES[1:], FONT_FAMILIES[:3]), 20
    ):
        snippets.append(
            f'<input type="text" placeholder="Type here..." '
            f'style="padding:{pad};border-radius:{radius};border:{border};font-family:{font};font-size:16px;" />'
        )

    random.shuffle(snippets)
    return snippets


def render_snippets(snippets, out_dir):
    """Render each snippet to a screenshot and return manifest entries."""
    screenshots_dir = os.path.join(out_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    manifest = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 512, "height": 512})

        for i, snippet in enumerate(snippets):
            full_html = (
                "<!DOCTYPE html>"
                "<html><head><meta charset='utf-8'>"
                "<style>body{margin:20px;background:#fff;}</style>"
                "</head><body>"
                f"{snippet}"
                "</body></html>"
            )
            page.set_content(full_html)
            page.wait_for_timeout(100)  # let rendering settle

            screenshot_path = os.path.join(screenshots_dir, f"{i:04d}.png")
            page.screenshot(path=screenshot_path)

            manifest.append({
                "id": i,
                "screenshot": screenshot_path,
                "html": snippet,
            })

            if (i + 1) % 25 == 0:
                print(f"Rendered {i + 1}/{len(snippets)} screenshots")

        browser.close()

    return manifest


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating HTML snippets...")
    snippets = generate_snippets()
    print(f"Generated {len(snippets)} snippets")

    print("Rendering screenshots...")
    manifest = render_snippets(snippets, out_dir)

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done! Saved {len(manifest)} pairs to {manifest_path}")


if __name__ == "__main__":
    main()
