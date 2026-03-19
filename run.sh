#!/bin/bash
set -e

# Load API key
source .env
export TINKER_API_KEY

echo "=== Step 1: Install dependencies ==="
pip install -r requirements.txt
pip install tinker-cookbook
playwright install chromium

echo ""
echo "=== Step 2: Generate dataset ==="
# Use web-sourced dataset (WebSight + Design2Code) instead of synthetic snippets
# To use simple synthetic snippets instead, run: python generate_dataset.py
python generate_dataset_web.py

echo ""
echo "=== Step 3: Train ==="
python train.py
