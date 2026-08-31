#!/usr/bin/env python
"""Render terminal-style frames for the demo video (1280x720).

The local ffmpeg has no libfreetype/libass, so drawtext/subtitles/ass are all
unavailable. Everything textual -- table frames, code frames, burned-in
subtitles -- is rendered here with Pillow and composited by ffmpeg's overlay.

Palette matches the recorded terminal segments (B1/B2/B3) so cuts between
rendered and recorded material do not jump.
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (12, 12, 12)
FG = (216, 216, 216)
DIM = (128, 128, 128)
GREEN = (126, 199, 108)
CYAN = (86, 182, 194)
YELLOW = (215, 186, 125)
RED = (233, 96, 76)
WHITE = (245, 245, 245)

MONO = "/System/Library/Fonts/Menlo.ttc"
SANS = "/System/Library/Fonts/Helvetica.ttc"


def mono(size, bold=False):
    return ImageFont.truetype(MONO, size, index=1 if bold else 0)


def sans(size, bold=True):
    return ImageFont.truetype(SANS, size, index=1 if bold else 0)


def new_frame():
    return Image.new("RGB", (W, H), BG)


def draw_rows(img, rows, x=40, y=40, size=17, leading=25):
    """rows: list of (text, colour) or (text, colour, bold)."""
    d = ImageDraw.Draw(img)
    for i, row in enumerate(rows):
        text, colour = row[0], row[1]
        bold = row[2] if len(row) > 2 else False
        d.text((x, y + i * leading), text, font=mono(size, bold), fill=colour)
    return img
