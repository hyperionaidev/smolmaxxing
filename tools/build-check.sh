#!/bin/bash
# Build targets in a build dir and FAIL LOUDLY: twice on 2026-11-07 a failed ninja was followed by benchmarks of the old binary.
# usage: tools/build-check.sh <build-dir> <target...>
set -u
D=$1; shift
cd "$(dirname "$0")/../third_party/llama.cpp/$D" || exit 2
if ! ninja "$@" > "${TMPDIR:-/tmp}/ninja-tv.log" 2>&1; then
  grep -E 'error|FAILED' "${TMPDIR:-/tmp}/ninja-tv.log" | cut -c1-200 | head -12
  echo "BUILD FAILED in $D"; exit 1
fi
echo "build ok: $D $*"
