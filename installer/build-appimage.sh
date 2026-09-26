#!/usr/bin/env bash
# Build the HCAD Linux release and package it as an AppImage.
#
#   ./installer/build-appimage.sh            # build + package
#   ./installer/build-appimage.sh --smoke    # also run ONE headless smoke test
#
# Output: dist/hcad-v<version>-linux-x86_64.AppImage
#
# Requirements (Ubuntu 24.04 / glibc 2.39 is the supported baseline):
#   rust (cargo), a C++ toolchain + cmake (for the Manifold kernel), `strip`,
#   `appimagetool`, and the Bevy dev libs (libasound2-dev libudev-dev
#   libwayland-dev libxkbcommon-dev vulkan-tools).
#
# RAM note: this is a heavy build. Parallelism is capped (JOBS=2) so a
# 16 GB machine doesn't run out of memory; raise JOBS if you have more.
set -euo pipefail

JOBS="${JOBS:-2}"
SMOKE=0
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
[[ -n "$VERSION" ]] || { echo "could not read version from Cargo.toml" >&2; exit 1; }
OUT="$ROOT/dist/hcad-v${VERSION}-linux-x86_64.AppImage"
APPDIR="$ROOT/target/AppDir"

for t in cargo strip appimagetool; do
  command -v "$t" >/dev/null || { echo "missing tool: $t" >&2; exit 1; }
done

echo "==> building hcad v$VERSION (release, -j $JOBS)"
cargo build --release -p hworks-app -j "$JOBS"
BIN="$ROOT/target/release/hcad"
[[ -x "$BIN" ]] || { echo "build did not produce $BIN" >&2; exit 1; }

echo "==> assembling AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp "$BIN" "$APPDIR/usr/bin/hcad"
strip "$APPDIR/usr/bin/hcad"

cat > "$APPDIR/hcad.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=HCAD
GenericName=Parametric CAD Modeler
Comment=A SolidWorks-style parametric CAD modeler
Exec=hcad
Icon=hcad
Categories=Graphics;Engineering;3DGraphics;
Terminal=false
StartupWMClass=hcad
DESKTOP
cp "$APPDIR/hcad.desktop" "$APPDIR/usr/share/applications/hcad.desktop"

cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
exec "$HERE/usr/bin/hcad" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

ICON="$ROOT/crates/hworks-app/assets/logo.png"
cp "$ICON" "$APPDIR/hcad.png"
cp "$ICON" "$APPDIR/usr/share/icons/hicolor/256x256/apps/hcad.png"

echo "==> packaging AppImage"
mkdir -p "$ROOT/dist"
ARCH=x86_64 appimagetool "$APPDIR" "$OUT"
echo "==> built: $OUT ($(du -h "$OUT" | cut -f1))"

if [[ "$SMOKE" -eq 1 ]]; then
  # One launch only: opens a window, renders 10 frames, exits. Needs Vulkan.
  echo "==> smoke test (single run)"
  "$OUT" --smoke 2>&1 | grep -E "Smoke test|panic" || true
fi
