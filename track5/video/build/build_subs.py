#!/usr/bin/env python
"""Render burned-in subtitles as transparent PNGs.

The local ffmpeg lacks libfreetype/libass, so drawtext/subtitles/ass cannot be
used. Each cue is rendered here and composited with overlay at build time.

Every cue is <= 12 words and holds >= 2.5s, per the script's pacing rule.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_frame import Image, ImageDraw, sans, W, H  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "video_assets", "subs")
os.makedirs(OUT, exist_ok=True)

SIZE = 30
PAD_X, PAD_Y = 26, 14
BOTTOM = 46          # distance from frame bottom to the plate
PLATE = (0, 0, 0, 216)
TEXT = (255, 255, 255, 255)

# B1/B2 terminal output runs to row ~716 of 720, so any plate over the video
# would cover a JSON record the cue is describing. compose.sh letterboxes those
# two segments into the top BAND_TOP px and the plate sits in the clear band
# below. Measured, not guessed: see the last-ink-row probe in the build log.
BAND_H = 74          # height of the clear subtitle band for letterboxed segs
LETTERBOXED = {"b1", "b2"}

# (segment, index, start-within-segment, duration, text)
CUES = [
    # A -- 16s: card 0-4, triptych 4-16
    ("a2", 1, 2.0, 4.0, "A detector that only works on clean images is not a detector."),
    ("a2", 2, 6.0, 3.0, "Every upload re-encodes. Every re-share resizes."),
    ("a2", 3, 9.2, 2.8, "These three look identical."),
    ("a2", 4, 12.0, 0.0, "The detector's evidence: 100%, 74%, 4%."),
    # B1 -- 19.2s
    ("b1", 1, 3.0, 4.0, "One script. A directory in, {image_path, pred} out."),
    ("b1", 2, 11.0, 4.0, "pred is a calibrated probability, not a raw logit."),
    ("b1", 3, 15.2, 0.0, "sigmoid((z + alpha) / T), frozen before inference."),
    # B2 -- 18.2s
    ("b2", 1, 1.0, 3.6, "The same images, re-encoded at JPEG q30 and 0.25x."),
    ("b2", 2, 5.0, 3.4, "Transformed by the repository's own eval_atoms, not a demo shim."),
    ("b2", 3, 11.4, 6.8, "Every generated image stays above 0.997. Every real photo below 0.019."),
    # B3 -- 6.5s
    ("b3", 1, 0.6, 0.0, "A decode failure is recorded, never silently scored 0.5."),
    # C -- 18s
    ("c1", 1, 0.8, 0.0, "The organiser's benchmark: DALL-E 3, unseen by our model."),
    ("c2", 1, 0.5, 0.0, "0.958 to 0.989 balanced accuracy across five conditions."),
    ("c3", 1, 0.5, 3.2, "Worst-case degradation from clean: 3.1 points."),
    ("c3", 2, 4.0, 0.0, "The whole DALL-E family, 64,482 images, is denylisted."),
    # D -- 22s
    ("d1", 1, 0.8, 4.0, "Three pass/fail criteria, written down before scoring."),
    ("d1", 2, 5.2, 0.0, "At that commit, section 4 was still empty."),
    ("d2", 1, 0.8, 3.6, "The scoring run landed 38 hours later."),
    ("d2", 2, 4.6, 0.0, "Verify it yourself: the git history is the receipt."),
    ("d3", 1, 0.5, 0.0, "Criterion 1 not triggered: 2 of 100. Both collapse rescaled."),
    # E -- 36s
    ("e1", 1, 0.8, 0.0, "Criterion 2 fired at 93.3 points, backwards."),
    ("e2", 1, 0.5, 0.0, "We guarded against the detector reading vendor watermarks."),
    ("e3", 1, 0.8, 5.0, "A watermark-reader would find the watermarked slice easy."),
    ("e3", 2, 6.0, 0.0, "Ours finds it nearly invisible. That refutes the mechanism."),
    ("e4", 1, 0.8, 5.0, "Not the transforms. Not unseen families: DALL-E 3 holds 0.9891."),
    ("e4", 2, 6.2, 0.0, "It is the 2026 generation of consumer endpoints."),
    # F -- 19s
    ("f1", 1, 0.8, 3.6, "Before training: a probe reading only the JPEG table scored 0.974."),
    ("f1", 2, 4.6, 0.0, "We equalised the containers. It fell to 0.500."),
    ("f2", 1, 0.5, 0.0, "Eleven limitations, ordered by how much each should move you."),
]


def render(text, path, in_band=False):
    f = sans(SIZE, bold=False)
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    tw = int(probe.textlength(text, font=f))
    bbox = f.getbbox(text)
    th = bbox[3] - bbox[1]

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if in_band:
        # centre the text in the clear band; no plate needed, nothing behind it
        py = H - BAND_H + (BAND_H - th) // 2 - bbox[1]
        d.text(((W - tw) // 2, py), text, font=f, fill=TEXT)
    else:
        pw, ph = tw + PAD_X * 2, th + PAD_Y * 2
        px = (W - pw) // 2
        py = H - BOTTOM - ph
        d.rounded_rectangle([px, py, px + pw, py + ph], radius=7, fill=PLATE)
        d.text((px + PAD_X, py + PAD_Y - bbox[1]), text, font=f, fill=TEXT)
    img.save(path)
    return tw


if __name__ == "__main__":
    manifest = []
    over = []
    for seg, idx, start, dur, text in CUES:
        words = len(text.split())
        name = f"{seg}_{idx}.png"
        render(text, os.path.join(OUT, name), in_band=seg in LETTERBOXED)
        if words > 12:
            over.append((name, words, text))
        manifest.append({"seg": seg, "file": name, "start": start,
                         "dur": dur, "words": words})
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"rendered {len(CUES)} cues -> {OUT}")
    if over:
        print("\nOVER 12 WORDS (script rule):")
        for n, w, t in over:
            print(f"  {n}  {w}w  {t}")
    else:
        print("all cues within the 12-word limit")
