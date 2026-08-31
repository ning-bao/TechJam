#!/bin/bash
# Composite subtitles onto each segment and concatenate the full cut.
# The local ffmpeg has no libass/libfreetype, so cues are pre-rendered PNGs
# layered with overlay+enable=between(t,...).
set -euo pipefail
cd "$(dirname "$0")/video_assets"

VENV=../.venv-img/bin/python
SEGS=(a1 a2 b1 b2 b3 c1 c2 c3 d1 d2 d3 e1 e2 e3 e4 f1 f2 f3)

mkdir -p final

# recorded terminal clips live in cut/, rendered ones in seg/
src_for() {
  case "$1" in
    b1) echo "cut/B1_cut.mp4" ;;
    b2) echo "cut/B2_cut.mp4" ;;
    b3) echo "cut/B3_cut.mp4" ;;
    *)  echo "seg/$1.mp4" ;;
  esac
}

# B1/B2 terminal output reaches row ~716 of 720, so a subtitle plate over the
# video would cover a JSON record. Letterbox them into the top 646 px and leave
# a clear 74 px band underneath for the cue.
BAND_H=74
letterboxed() { case "$1" in b1|b2) return 0 ;; *) return 1 ;; esac; }

for s in "${SEGS[@]}"; do
  src=$(src_for "$s")
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$src")
  # collect cues for this segment from the manifest
  cues=()
  while IFS= read -r line; do
    [ -n "$line" ] && cues+=("$line")
  done < <($VENV - "$s" "$dur" <<'PY'
import json, sys, os
seg, dur = sys.argv[1], float(sys.argv[2])
m = json.load(open("subs/manifest.json"))
for c in m:
    if c["seg"] != seg:
        continue
    end = dur if c["dur"] == 0.0 else c["start"] + c["dur"]
    print(f'{c["file"]}\t{c["start"]:.2f}\t{min(end, dur):.2f}')
PY
)
  if [ ${#cues[@]} -eq 0 ]; then
    ffmpeg -v error -i "$src" -an -c:v libx264 -crf 18 -preset slow \
           -pix_fmt yuv420p -r 30 "final/$s.mp4" -y
    printf "%-4s %6.2fs  (no cues)\n" "$s" "$dur"
    continue
  fi
  inputs=(-i "$src"); filter=""; prev="0:v"; n=1
  if letterboxed "$s"; then
    vh=$((720 - BAND_H))
    filter+="[0:v]scale=-2:${vh}:flags=lanczos,pad=1280:720:(ow-iw)/2:0:color=0x000000[lb];"
    prev="lb"
  fi
  for c in "${cues[@]}"; do
    IFS=$'\t' read -r file st en <<<"$c"
    inputs+=(-i "subs/$file")
    filter+="[$prev][$n:v]overlay=0:0:enable='between(t,$st,$en)'[v$n];"
    prev="v$n"; n=$((n+1))
  done
  filter="${filter%;}"
  ffmpeg -v error "${inputs[@]}" -filter_complex "$filter" -map "[$prev]" \
         -an -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -r 30 \
         "final/$s.mp4" -y
  printf "%-4s %6.2fs  %d cues\n" "$s" "$dur" "${#cues[@]}"
done

# concatenate
: > final/list.txt
for s in "${SEGS[@]}"; do echo "file '$s.mp4'" >> final/list.txt; done
ffmpeg -v error -f concat -safe 0 -i final/list.txt -c copy final/demo_full.mp4 -y
echo
echo "=== full cut ==="
ffprobe -v error -show_entries format=duration:stream=width,height,r_frame_rate \
        -of default=nw=1 final/demo_full.mp4
