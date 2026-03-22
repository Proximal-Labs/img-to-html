"""
Fast RL training: screenshot + 1 action → interactive HTML.

All throughput optimizations applied:
- Parallel sampling: fire ALL futures at once, don't wait sequentially
- Train-as-you-go: forward_backward on minibatches as rollouts complete
- Small images: 384x216 for prompts/feedback, minimize token count
- 1 action step only: initial page + 1 click result
- No waits: 50ms instead of 500ms
- Overlap forward_backward + optim_step across Tinker clock cycles

Task: model sees initial screenshot + post-click screenshot,
generates HTML, we render + click + compare SSIM.
If SSIM < 0.8: show ref+gen+diff (small), model analyzes + fixes.
Final reward = SSIM on both states.
"""

import json
import logging
import os
import random
import sys
import time
import io
from concurrent.futures import as_completed, Future

import numpy as np
import torch
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from PIL import Image
from playwright.sync_api import sync_playwright
from tqdm import tqdm
from transformers import AutoImageProcessor
from skimage.metrics import structural_similarity as ssim_fn

from tinker_cookbook import renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    MODEL, LORA_RANK, RENDERER_NAME, LR, KL_BETA,
    PPO_CLIP_LOW, PPO_CLIP_HIGH, LOG_DIR,
)
from reward import render_html, extract_html_from_response, make_diff_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Speed-optimized defaults ─────────────────────────────────────────────────
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", 2))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", 0))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 5))
TOKENS_PER_TURN = int(os.environ.get("TOKENS_PER_TURN", 4096))
N_TASKS = int(os.environ.get("N_TASKS", 500))
MINIBATCH_SIZE = int(os.environ.get("MINIBATCH_SIZE", 8))  # train every N datums

IMG_SIZE = (384, 216)  # Aggressive downsize for speed
VIEWPORT = {"width": 1280, "height": 720}
RENDER_WAIT_MS = 50

SYSTEM_PROMPT = (
    "You generate interactive HTML/CSS/JS from website screenshots. "
    "You will see the initial page and what happens after one user action. "
    "Generate HTML that reproduces both states. "
    "Wrap code in ```html ... ```."
)


def take_screenshot_small(page) -> np.ndarray:
    raw = np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
    return np.array(Image.fromarray(raw).resize(IMG_SIZE))


def compute_ssim(a, b):
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0])))
    return float(ssim_fn(a, b, channel_axis=2, data_range=255))


def load_tasks(n: int, seed: int = 42) -> list[dict]:
    """Load Mind2Web tasks, keep only those with 2+ actions."""
    from datasets import load_dataset

    logger.info("Loading Mind2Web (streaming)...")
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="train", streaming=True)

    tasks = {}
    for i, row in enumerate(ds):
        if len(tasks) >= n * 3:
            break
        if i > 10000:
            break

        ann_id = row["annotation_id"]
        if ann_id not in tasks:
            tasks[ann_id] = {"task": row["confirmed_task"], "website": row["website"], "actions": []}

        op = json.loads(row["operation"]) if isinstance(row["operation"], str) else row["operation"]
        pos = json.loads(row["pos_candidates"][0]) if row["pos_candidates"] else {}
        if isinstance(pos, str):
            pos = json.loads(pos)
        attrs = pos.get("attributes", "{}") if isinstance(pos, dict) else "{}"
        if isinstance(attrs, str):
            attrs = json.loads(attrs)

        screenshot = row["screenshot"]
        if screenshot:
            viewport_img = screenshot.crop((0, 0, min(screenshot.width, 1280), min(screenshot.height, 720)))
        else:
            viewport_img = None

        repr_str = row["target_action_reprs"]
        action_text = ""
        if "->" in repr_str:
            action_text = repr_str.split("->")[0].strip()
            if "]" in action_text:
                action_text = action_text.split("]", 1)[1].strip()

        tasks[ann_id]["actions"].append({
            "op": op.get("op", "CLICK"),
            "value": op.get("value", ""),
            "selector": f"#{attrs['id']}" if attrs.get("id") else None,
            "text": action_text,
            "repr": repr_str,
            "screenshot": viewport_img,
        })

    valid = [t for t in tasks.values() if len(t["actions"]) >= 2 and t["actions"][0]["screenshot"] and t["actions"][1]["screenshot"]]
    random.seed(seed)
    random.shuffle(valid)
    selected = valid[:n]
    logger.info(f"  {len(selected)} tasks with 2+ screenshots")
    return selected


def build_prompt(task, renderer):
    """Build prompt: initial screenshot + action + result screenshot. All small."""
    a0 = task["actions"][0]
    a1 = task["actions"][1]

    img0 = a0["screenshot"].resize(IMG_SIZE)
    img1 = a1["screenshot"].resize(IMG_SIZE)

    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Initial page:"},
            {"type": "image", "image": img0},
            {"type": "text", "text": f"After action: {a1['repr'][:60]}"},
            {"type": "image", "image": img1},
            {"type": "text", "text": "Generate HTML/CSS/JS that reproduces both states. Wrap in ```html ... ```."},
        ]},
    ]
    return renderer.build_generation_prompt(convo)


def try_click(page, action):
    """Try to click an element by selector, text, or bbox."""
    if action.get("selector"):
        try:
            page.click(action["selector"], timeout=1000)
            return True
        except Exception:
            pass
    if action.get("text"):
        try:
            page.get_by_text(action["text"], exact=False).first.click(timeout=1000)
            return True
        except Exception:
            pass
        try:
            page.get_by_role("button", name=action["text"]).first.click(timeout=1000)
            return True
        except Exception:
            pass
        try:
            page.get_by_role("link", name=action["text"]).first.click(timeout=1000)
            return True
        except Exception:
            pass
    return False


def compute_reward(page, html, task):
    """Render HTML, take initial screenshot, click, take post-click screenshot, compare SSIM."""
    if html is None:
        return -1.0, None, None

    try:
        render_html(page, html)
        page.wait_for_timeout(RENDER_WAIT_MS)
    except Exception:
        return -1.0, None, None

    # Initial state
    gen_img0 = take_screenshot_small(page)
    ref_img0 = np.array(task["actions"][0]["screenshot"].resize(IMG_SIZE))
    ssim0 = compute_ssim(ref_img0, gen_img0)

    # Click first action
    a1 = task["actions"][1]
    if a1["op"] == "CLICK":
        try_click(page, a1)
        page.wait_for_timeout(RENDER_WAIT_MS)

    # Post-click state
    gen_img1 = take_screenshot_small(page)
    ref_img1 = np.array(a1["screenshot"].resize(IMG_SIZE))
    ssim1 = compute_ssim(ref_img1, gen_img1)

    # Average SSIM across both states
    reward = (ssim0 + ssim1) / 2.0
    reward = 2.0 * reward - 1.0  # Scale to [-1, 1]

    return float(reward), gen_img0, gen_img1


def build_feedback(task, gen_img0, gen_img1, reward):
    """Build analyze feedback with small images."""
    ref_img0 = np.array(task["actions"][0]["screenshot"].resize(IMG_SIZE))
    ref_img1 = np.array(task["actions"][1]["screenshot"].resize(IMG_SIZE))

    diff0 = make_diff_image(ref_img0, gen_img0, threshold=25)
    ssim0 = compute_ssim(ref_img0, gen_img0)
    ssim1 = compute_ssim(ref_img1, gen_img1)

    content = [
        {"type": "text", "text": f"Initial page SSIM: {ssim0:.0%}, Post-action SSIM: {ssim1:.0%}\n\nInitial diff (red=wrong):"},
        {"type": "image", "image": Image.fromarray(diff0)},
        {"type": "text", "text": "Your initial page:"},
        {"type": "image", "image": Image.fromarray(gen_img0)},
        {"type": "text", "text": "Target initial:"},
        {"type": "image", "image": Image.fromarray(ref_img0)},
    ]

    if ssim1 < ssim0:  # Post-click is worse
        diff1 = make_diff_image(ref_img1, gen_img1, threshold=25)
        content.extend([
            {"type": "text", "text": f"\nPost-action diff:"},
            {"type": "image", "image": Image.fromarray(diff1)},
        ])

    content.append({"type": "text", "text": "List top 3 fixes. Be specific."})
    return content


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    dataset = load_tasks(N_TASKS)
    n_batches = len(dataset) // BATCH_SIZE
    if MAX_BATCHES > 0:
        n_batches = min(n_batches, MAX_BATCHES)
    logger.info(f"{len(dataset)} tasks, {n_batches} batches (BS={BATCH_SIZE}, GS={GROUP_SIZE})")

    tokenizer = get_tokenizer(MODEL)
    image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer, image_processor=image_processor)

    service_client = tinker.ServiceClient()
    resume_path = os.environ.get("RESUME_FROM")
    if resume_path:
        logger.info(f"Resuming: {resume_path}")
        training_client = service_client.create_training_client_from_state_with_optimizer(resume_path)
    else:
        training_client = service_client.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)

    sampling_params = types.SamplingParams(
        max_tokens=TOKENS_PER_TURN, stop=renderer.get_stop_sequences(), temperature=0.7,
    )
    adam_params = types.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)

    metrics_path = os.path.join(LOG_DIR, "metrics_fast.jsonl")
    metrics_file = open(metrics_path, "a")

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    pages = [browser.new_page(viewport=VIEWPORT) for _ in range(BATCH_SIZE)]

    logger.info(f"Config: BS={BATCH_SIZE} GS={GROUP_SIZE} KL={KL_BETA} TOKENS={TOKENS_PER_TURN} IMG={IMG_SIZE}")

    for batch_idx in range(n_batches):
        t_start = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        if SAVE_EVERY > 0 and batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            logger.info(f"Checkpoint {batch_idx}...")
            training_client.save_state(name=f"checkpoint-{batch_idx:04d}").result()

        sampling_client = training_client.save_weights_and_get_sampling_client()

        # ── PHASE 1: Fire ALL turn-1 samples in parallel ─────────────
        prompts = []
        turn1_futures = []
        for task in batch:
            prompt = build_prompt(task, renderer)
            prompts.append(prompt)
            turn1_futures.append(
                sampling_client.sample(prompt=prompt, num_samples=GROUP_SIZE, sampling_params=sampling_params)
            )

        # ── PHASE 2: Process results + train-as-you-go ───────────────
        all_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_kl: list[float] = []
        fwd_bwd_futures: list = []  # Track in-flight training

        for idx in tqdm(range(len(batch)), desc=f"B{batch_idx}"):
            task = batch[idx]
            page = pages[idx % len(pages)]
            prompt = prompts[idx]

            turn1_result = turn1_futures[idx].result()

            rewards_G = []
            tokens_G = []
            logprobs_G = []

            # Fire turn-2 analyze+fix for all rollouts that need it
            turn2_data = []  # (g, all_tokens, all_logprobs, html, gen_img0, gen_img1, reward)

            for g, seq in enumerate(turn1_result.sequences):
                all_tokens = list(seq.tokens)
                all_logprobs = list(seq.logprobs)

                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg)
                html = extract_html_from_response(content)

                reward, gen_img0, gen_img1 = compute_reward(page, html, task)

                if reward < 0.8 and html is not None and gen_img0 is not None:
                    # Need analyze+fix
                    turn2_data.append((g, all_tokens, all_logprobs, content, html, gen_img0, gen_img1, reward))
                else:
                    tokens_G.append(all_tokens)
                    logprobs_G.append(all_logprobs)
                    kl = -sum(all_logprobs) / len(all_logprobs) if all_logprobs else 0.0
                    rewards_G.append(reward - KL_BETA * kl)
                    batch_kl.append(kl)

            # Fire all analyze calls in parallel
            analyze_futures = []
            for g, all_tokens, all_logprobs, content, html, gen_img0, gen_img1, reward in turn2_data:
                feedback = build_feedback(task, gen_img0, gen_img1, reward)
                convo = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompts[idx]._chunks[1:]},  # reuse prompt content
                ]
                # Simpler: rebuild from scratch
                convo_msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Initial page:"},
                        {"type": "image", "image": task["actions"][0]["screenshot"].resize(IMG_SIZE)},
                        {"type": "text", "text": f"After action: {task['actions'][1]['repr'][:60]}"},
                        {"type": "image", "image": task["actions"][1]["screenshot"].resize(IMG_SIZE)},
                        {"type": "text", "text": "Generate HTML/CSS/JS. Wrap in ```html ... ```."},
                    ]},
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": feedback},
                ]
                analyze_prompt = renderer.build_generation_prompt(convo_msgs)
                analyze_futures.append((
                    g, all_tokens, all_logprobs, reward,
                    sampling_client.sample(prompt=analyze_prompt, num_samples=1, sampling_params=sampling_params),
                    convo_msgs,
                ))

            # Collect analyze results, fire fix calls
            fix_futures = []
            for g, all_tokens, all_logprobs, reward, analyze_future, convo_msgs in analyze_futures:
                analyze_result = analyze_future.result()
                analyze_seq = analyze_result.sequences[0]
                all_tokens.extend(analyze_seq.tokens)
                all_logprobs.extend(analyze_seq.logprobs)

                analyze_msg, _ = renderer.parse_response(analyze_seq.tokens)
                analysis = get_text_content(analyze_msg)

                convo_msgs.append({"role": "assistant", "content": analysis})
                convo_msgs.append({"role": "user", "content": "Fix ALL issues. Output complete HTML in ```html ... ```."})
                fix_prompt = renderer.build_generation_prompt(convo_msgs)

                fix_futures.append((
                    g, all_tokens, all_logprobs,
                    sampling_client.sample(prompt=fix_prompt, num_samples=1, sampling_params=sampling_params),
                ))

            # Collect fix results
            for g, all_tokens, all_logprobs, fix_future in fix_futures:
                fix_result = fix_future.result()
                fix_seq = fix_result.sequences[0]
                all_tokens.extend(fix_seq.tokens)
                all_logprobs.extend(fix_seq.logprobs)

                parsed_fix, _ = renderer.parse_response(fix_seq.tokens)
                fix_html = extract_html_from_response(get_text_content(parsed_fix))
                reward, _, _ = compute_reward(page, fix_html, task)

                tokens_G.append(all_tokens)
                logprobs_G.append(all_logprobs)
                kl = -sum(all_logprobs) / len(all_logprobs) if all_logprobs else 0.0
                rewards_G.append(reward - KL_BETA * kl)
                batch_kl.append(kl)

            # GRPO advantages
            if not rewards_G:
                continue
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            batch_rewards.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                continue

            for tokens, logprobs, advantage in zip(tokens_G, logprobs_G, advantages_G):
                if not tokens:
                    continue
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

                assert model_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)

                all_datums.append(types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                ))

            # ── Train-as-you-go: submit minibatch if enough datums ────
            if len(all_datums) >= MINIBATCH_SIZE:
                fwd_bwd_futures.append(
                    training_client.forward_backward(
                        all_datums[:MINIBATCH_SIZE], loss_fn="ppo",
                        loss_fn_config={"clip_low_threshold": PPO_CLIP_LOW, "clip_high_threshold": PPO_CLIP_HIGH},
                    )
                )
                all_datums = all_datums[MINIBATCH_SIZE:]

        # ── PHASE 3: Train remaining + optim step ────────────────────
        if all_datums:
            fwd_bwd_futures.append(
                training_client.forward_backward(
                    all_datums, loss_fn="ppo",
                    loss_fn_config={"clip_low_threshold": PPO_CLIP_LOW, "clip_high_threshold": PPO_CLIP_HIGH},
                )
            )

        if not fwd_bwd_futures:
            logger.warning(f"Batch {batch_idx}: no datums")
            continue

        # Wait for all fwd_bwd, then optim
        optim_future = training_client.optim_step(adam_params)
        for f in fwd_bwd_futures:
            f.result()
        optim_future.result()

        elapsed = time.time() - t_start
        mean_r = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        mean_kl = sum(batch_kl) / len(batch_kl) if batch_kl else 0.0
        logger.info(f"Batch {batch_idx}/{n_batches}: reward={mean_r:.3f} kl={mean_kl:.3f} datums={len(fwd_bwd_futures)} time={elapsed:.0f}s")
        metrics_file.write(json.dumps({"batch": batch_idx, "reward": round(mean_r, 4), "kl": round(mean_kl, 4), "time": round(elapsed, 1)}) + "\n")
        metrics_file.flush()
        fwd_bwd_futures = []

    metrics_file.close()

    logger.info("Saving final...")
    training_client.save_state(name="final").result()
    result = training_client.save_weights_for_sampler(name="prox-fast-final").result()
    logger.info(f"Saved: {result.path}")
    with open(os.path.join(LOG_DIR, "model_path_fast.txt"), "w") as f:
        f.write(result.path)

    browser.close()
    pw.stop()
    logger.info("Done!")


if __name__ == "__main__":
    main()
