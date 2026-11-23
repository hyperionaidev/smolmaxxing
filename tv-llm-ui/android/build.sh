#!/bin/bash
# Build the SmolChat TV app without Gradle: javac -> d8 -> aapt2 -> add dex, binaries and page -> zipalign -> apksigner.
# Needs JDK 17 and, in third_party/android-sdk (git-ignored):
#   sdkmanager --sdk_root=third_party/android-sdk "build-tools;34.0.0" "platforms;android-34"
# Packs bin/patched25/llama-server and bin/tv-watch (docs/tv-box-setup.md, step 1) as lib*.so: an app may only
# execute files from its native library directory. Output: tv-llm-ui/android/build/smolchat.apk
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
SDK=${ANDROID_SDK:-$ROOT/third_party/android-sdk}
BT=$SDK/build-tools/34.0.0
JAR=$SDK/platforms/android-34/android.jar
KS=${KEYSTORE:-$SDK/smolchat.keystore}
O=build
# The Xiaomi box runs 32-bit Android (abilist armeabi-v7a) on a 64-bit kernel: apps are armeabi-v7a, but the kernel runs
# the static aarch64 binaries fine (that is how the adb-started server works). The package manager copies lib/<abi>/*.so
# without checking the ELF class, so the aarch64 binaries go in the 32-bit folder. ABI=arm64-v8a for a 64-bit Android.
ABI=${ABI:-armeabi-v7a}
for f in "$BT/aapt2" "$JAR" "$ROOT/bin/patched25/llama-server" "$ROOT/bin/tv-watch"; do
  [ -e "$f" ] || { echo "missing $f (see the header of this script)"; exit 1; }
done
rm -rf $O; mkdir -p $O/res/drawable $O/classes $O/apk/lib/$ABI $O/apk/assets/ui

# launcher art: the repo banner fitted into the 320x180 TV banner and a square icon
sips -s format png -Z 320 "$ROOT/assets/banner.png" --out $O/res/drawable/banner.png >/dev/null
sips --padToHeightWidth 180 320 --padColor 0B0B10 $O/res/drawable/banner.png >/dev/null 2>&1
sips -s format png -Z 192 "$ROOT/assets/banner.png" --out $O/res/drawable/icon.png >/dev/null
sips --padToHeightWidth 192 192 --padColor 0B0B10 $O/res/drawable/icon.png >/dev/null 2>&1

"$BT/aapt2" compile -o $O/res.zip --dir $O/res
"$BT/aapt2" link -o $O/base.apk -I "$JAR" --manifest AndroidManifest.xml $O/res.zip

javac -nowarn -Xlint:-options -source 11 -target 11 -cp "$JAR" -d $O/classes $(find src -name '*.java')
"$BT/d8" --release --min-api 28 --lib "$JAR" --output $O/apk $(find $O/classes -name '*.class')

cp "$ROOT/bin/patched25/llama-server" $O/apk/lib/$ABI/libllamaserver.so
cp "$ROOT/bin/tv-watch" $O/apk/lib/$ABI/libtvwatch.so
cp ../index.html ../config.json $O/apk/assets/ui/
(cd $O/apk && zip -qr ../base.apk classes.dex lib assets)

"$BT/zipalign" -f -p 4 $O/base.apk $O/aligned.apk
if [ ! -f "$KS" ]; then
  keytool -genkeypair -keystore "$KS" -storepass smolchat -keypass smolchat -alias smolchat \
    -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=SmolChat" >/dev/null 2>&1
fi
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:smolchat --key-pass pass:smolchat --out $O/smolchat.apk $O/aligned.apk
ls -la $O/smolchat.apk
