#!/bin/sh
# Renders tools/banner/banner.html to assets/banner.png (2560x720) with headless Chrome.
set -e
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files \
  --force-device-scale-factor=2 --window-size=1280,360 --virtual-time-budget=5000 \
  --screenshot="$PWD/../../assets/banner.png" "file://$PWD/banner.html"
