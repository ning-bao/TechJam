#!/usr/bin/env python
"""Render every non-recorded frame for the demo video (1280x720).

All numbers here were verified against the repository before rendering:
  C  robustness_summary.md  protected benchmark table
  D  git show / git log     real command output, run live
  E  error_analysis.md      set B slice table
  F  README.md probes + track5/KNOWN_LIMITATIONS.md headings
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_frame import (  # noqa: E402
    ImageDraw, W, H, BG, FG, DIM, GREEN, CYAN, YELLOW, RED, WHITE,
    mono, sans, new_frame, draw_rows,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "video_assets", "frames")
os.makedirs(OUT, exist_ok=True)

RULE = (52, 52, 52)
PROMPT_USER = (126, 199, 108)
PROMPT_PATH = (215, 186, 125)


def header(img, path):
    d = ImageDraw.Draw(img)
    d.text((44, 32), path, font=mono(15), fill=DIM)
    d.line([(44, 60), (1236, 60)], fill=RULE, width=1)


def save(img, name):
    img.save(os.path.join(OUT, name))
    print("  ", name)


def prompt_line(d, y, cmd, size=17):
    """$ cmd  with a green prompt, mimicking the recorded MINGW64 shell."""
    d.text((44, y), "$", font=mono(size, True), fill=PROMPT_USER)
    d.text((44 + 20, y), cmd, font=mono(size), fill=FG)


# ---------------------------------------------------------------- C segment
def build_C():
    print("C segment")
    rows_base = [
        ("Protected benchmark — COCO val2017 reals (5,000) vs DALL·E 3 (8,843)", DIM),
        ("The entire DALL·E family — 64,482 images — is denylisted from training.", DIM),
        ("", FG),
        ("condition       bAcc      95% CI            AUROC    FPR@95TPR   ECE", CYAN, True),
        ("─" * 68, RULE),
    ]
    data = [
        ("clean           0.9891    0.9876–0.9907     0.9998   0.000       0.013", "clean"),
        ("jpeg_30         0.9643    0.9612–0.9672     0.9931   0.024       0.025", None),
        ("blur_20         0.9667    0.9639–0.9694     0.9967   0.012       0.035", None),
        ("resize_025      0.9584    0.9548–0.9617     0.9909   0.032       0.029", "resize"),
        ("noise_010       0.9614    0.9585–0.9647     0.9938   0.026       0.032", None),
    ]
    tail = [
        ("─" * 68, RULE),
        ("mean bAcc 0.968 · worst case 0.9584 · max degradation 3.1 points", GREEN, True),
    ]

    # C1 plain table, C2 clean highlighted, C3 resize_025 highlighted
    for idx, hl in enumerate((None, "clean", "resize")):
        img = new_frame()
        header(img, "track5/reports/robustness_summary.md")
        rows = list(rows_base)
        for text, tag in data:
            if tag is not None and tag == hl:
                rows.append((text, WHITE, True))
            elif tag == "resize":
                rows.append((text, YELLOW, True))
            elif tag == "clean":
                rows.append((text, WHITE, True))
            else:
                rows.append((text, FG))
        rows += tail
        draw_rows(img, rows, x=44, y=88, size=18, leading=30)
        d = ImageDraw.Draw(img)
        if hl == "clean":
            d.line([(40, 232), (40, 260)], fill=RED, width=4)
        elif hl == "resize":
            d.line([(40, 322), (40, 350)], fill=RED, width=4)
        save(img, f"C{idx + 1}.png")


# ---------------------------------------------------------------- D segment
CMD1 = ("git show e6c8349:track5/reports/error_analysis.md \\\n"
        "    | grep -A13 '^## 4\\.'")
CMD2 = "git log --format='%h %ai %s' -4 -- track5/reports/error_analysis.md"


def _read(path):
    with open(path) as fh:
        return [ln.rstrip("\n") for ln in fh]


def build_D(cmd1_out, cmd2_out):
    print("D segment")

    # D1: criteria written down, § 4 still empty at that commit
    img = new_frame()
    d = ImageDraw.Draw(img)
    d.text((44, 30), "# the moment the criteria were written, § 4 was still empty",
           font=mono(15), fill=DIM)
    prompt_line(d, 58, "git show e6c8349:track5/reports/error_analysis.md \\")
    d.text((44 + 20, 84), "    | grep -A13 '^## 4.'", font=mono(17), fill=FG)
    rows = []
    for ln in cmd1_out[:13]:
        if ln.startswith("## 4."):
            rows.append((ln, CYAN, True))
        elif "Acceptance criteria decided in advance" in ln:
            rows.append((ln, GREEN, True))
        else:
            rows.append((ln, FG))
    draw_rows(img, rows, x=44, y=124, size=17, leading=26)
    save(img, "D1.png")

    # D2: the scoring run landed 38 hours later
    img = new_frame()
    d = ImageDraw.Draw(img)
    d.text((44, 30), "# ... and the scoring run landed 38 hours later",
           font=mono(15), fill=DIM)
    prompt_line(d, 58, CMD2)
    rows = []
    for i, ln in enumerate(cmd2_out):
        sha, date, tm, tz = ln.split(" ", 3)[0], *ln.split(" ")[1:4]
        subj = ln.split(" ", 4)[4]
        first_or_last = (i == 0 or i == len(cmd2_out) - 1)
        rows.append((f"{sha} {date} {tm}", YELLOW if first_or_last else FG,
                     first_or_last))
        rows.append((f"        {subj[:74]}", FG if first_or_last else DIM))
    draw_rows(img, rows, x=44, y=112, size=17, leading=26)
    d = ImageDraw.Draw(img)
    d.text((44, 112 + len(rows) * 26 + 24),
           "38 hours between the criteria and the scores. Verify it yourself.",
           font=mono(18, True), fill=GREEN)
    save(img, "D2.png")

    # D3: set A false positives collapse when rescaled
    img = new_frame()
    header(img, "track5/reports/error_analysis.md  § 4")
    rows = [
        ("Criterion 1 — set A false positives at the frozen τ", CYAN, True),
        ("Not triggered: 2/100 native, 1/100 downscaled.", GREEN, True),
        ("", FG),
        ("image                             native    downscaled", CYAN, True),
        ("─" * 56, RULE),
        ("minimal_or_incoherent/027.jpg     0.8012    0.0041", WHITE, True),
        ("minimal_or_incoherent/020.jpg     0.0438    0.9558", YELLOW, True),
        ("─" * 56, RULE),
        ("They were reporting resolution, not photography.", FG),
        ("020.jpg is the counterexample kept: benign native, 0.956 downscaled.", DIM),
    ]
    draw_rows(img, rows, x=44, y=92, size=18, leading=31)
    save(img, "D3.png")


# ---------------------------------------------------------------- E segment
def build_E():
    print("E segment")
    slice_rows = [
        ("flux_dev  (seen family, no watermark)      25/25    0.9993", "flux"),
        ("gpt-image-2  (Azure)                        0/30    0.0038", "gpt"),
        ("gemini-3-pro-image                          3/25    0.0076", None),
        ("gemini-3.1-flash-image                      2/20    0.0065", None),
    ]
    for idx, hl in enumerate((None, "flux", "gpt")):
        img = new_frame()
        header(img, "track5/reports/error_analysis.md  § 4  —  set B slices")
        rows = [
            ("Criterion 2 fired at 93.3 points — in the reverse direction.", CYAN, True),
            ("", FG),
            ("set B slice                            recall @ τ    median", CYAN, True),
            ("─" * 60, RULE),
        ]
        for text, tag in slice_rows:
            if tag == hl:
                rows.append((text, WHITE if tag == "flux" else RED, True))
            elif tag in ("flux", "gpt"):
                rows.append((text, YELLOW, True))
            else:
                rows.append((text, FG))
        rows += [
            ("─" * 60, RULE),
            ("", FG),
            ("A watermark-reader would find the watermarked slice easy.", DIM),
            ("Ours finds it nearly invisible — that refutes the mechanism.", GREEN, True),
        ]
        draw_rows(img, rows, x=44, y=92, size=18, leading=31)
        save(img, f"E{idx + 1}.png")

    # E4: limitation item 11
    img = new_frame()
    header(img, "track5/KNOWN_LIMITATIONS.md  —  item 11")
    rows = [
        ("11. 2026 consumer generators evade the frozen", CYAN, True),
        ("    operating point", CYAN, True),
        ("", FG),
        ("Flux control (seen family):        25/25   median 0.9993", FG),
        ("2026 consumer-endpoint slice:       5/75   median 0.004–0.008", RED, True),
        ("", FG),
        ("Confidently wrong, so ranking fails along with the threshold.", FG),
        ("", FG),
        ("Not watermark-reading. Not unseen-family fragility alone —", DIM),
        ("DALL·E 3 is unseen and holds 0.9891.", DIM),
        ("It is the 2026 generation of consumer endpoints specifically.", GREEN, True),
    ]
    draw_rows(img, rows, x=44, y=92, size=18, leading=31)
    save(img, "E4.png")


# ---------------------------------------------------------------- F segment
def build_F():
    print("F segment")
    # F1: shortcut probe table
    img = new_frame()
    header(img, "README.md  —  shortcut probes, run before training")
    rows = [
        ("Training was gated on all four probes scoring below 0.60.", DIM),
        ("", FG),
        ("probe            raw corpus    normalized    size-matched", CYAN, True),
        ("─" * 58, RULE),
        ("JPEG quality        0.974         0.496         0.500", YELLOW, True),
        ("dimensions          0.732         0.500         0.500", FG),
        ("file size           0.589         0.620         0.497", FG),
        ("─" * 58, RULE),
        ("", FG),
        ("A probe reading only the JPEG table scored 0.974.", FG),
        ("We equalised the containers. It fell to 0.500.", GREEN, True),
    ]
    draw_rows(img, rows, x=44, y=92, size=18, leading=31)
    save(img, "F1.png")

    # F2: the full limitations list
    img = new_frame()
    header(img, "track5/KNOWN_LIMITATIONS.md")
    items = [
        "1.  The 448 px crop excludes four generator families from training",
        "2.  A second training epoch made the model worse, and there is a bug",
        "3.  Calibration does not transfer to unseen generators",
        "4.  Protected-set discipline was overrun by two reads",
        "5.  The hard-case evaluation sets are not comparable at clean",
        "6.  Per-item scores are not persisted",
        "7.  Single-platform sourcing in the curated real set",
        "8.  Watermark status is what the file declares, not what was detected",
        "9.  Duplicate rows in the protected benchmark",
        "10. Smaller things",
        "11. 2026 consumer generators evade the frozen operating point",
    ]
    rows = [("Eleven limitations, ordered by how much each should move", DIM),
            ("your confidence.", DIM), ("", FG)]
    for it in items:
        emph = it.startswith(("4.", "11."))
        rows.append((it, YELLOW if emph else FG, emph))
    draw_rows(img, rows, x=44, y=88, size=17, leading=28)
    save(img, "F2.png")


if __name__ == "__main__":
    build_C()
    build_D(_read("/tmp/d_cmd1.txt"), _read("/tmp/d_cmd2.txt"))
    build_E()
    build_F()
    print("done ->", OUT)
