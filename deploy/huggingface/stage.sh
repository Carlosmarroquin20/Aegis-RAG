#!/usr/bin/env bash
# Assembles the files a Hugging Face Space needs to build the demo image.
#
#   deploy/huggingface/stage.sh [output-dir]      (default: build/hf-space)
#
# A Space is its own git repository whose root holds the Dockerfile and a
# README.md with the Space's settings in its front matter. The directory made
# here is that root: the committed sources the image needs (git archive, so
# local node_modules or .env files can never be included), with the demo
# Dockerfile and README promoted to the top. The paths inside stay the same
# as in this repository, so one Dockerfile builds in both places.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
out=${1:-"$repo_root/build/hf-space"}

rm -rf "$out"
mkdir -p "$out"

git -C "$repo_root" archive --format=tar HEAD \
    pyproject.toml uv.lock src frontend \
    scripts/seed_documents.py examples/atencion-cliente deploy/huggingface \
    | tar -x -C "$out"

cp "$out/deploy/huggingface/Dockerfile" "$out/Dockerfile"
# Also the package's readme: pyproject.toml names README.md and hatchling
# reads it while building, so the Space README doubles as that file.
cp "$out/deploy/huggingface/README.md" "$out/README.md"

echo "Staged $(git -C "$repo_root" rev-parse --short HEAD) into $out"
