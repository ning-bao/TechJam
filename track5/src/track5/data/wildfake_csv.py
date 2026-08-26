"""WildFake label-CSV index (data/raw/WildFake/label_csv_files/*.csv).

The CSVs are the authoritative label/family/advanced-tier list for the whole
corpus and are already on disk, while the image archives are still downloading.
So we index from the CSVs and resolve each row to a zip member only if that
archive is present — a partially downloaded corpus yields a smaller manifest,
never a wrong one.

Constraint C2 lives here: the entire DALL·E family (dalle2.csv Typical +
dalle3.csv Advanced) is protected by *path*, which works even though none of the
DALLE archives are downloaded, so no code path can turn them into training data.
"""

import csv
import sys
import zipfile
from collections import namedtuple
from pathlib import Path

CSV_SUBDIR = "WildFake/label_csv_files"
IMAGES_SUBDIR = "WildFake/Images"

# CSV "Architecture" -> INTERFACES §4 generator_family vocabulary.
ARCH_TO_FAMILY = {
    "SD": "sd", "Midjourney": "mj", "DALLE": "dalle", "ADM": "adm",
    "DDPM": "ddpm", "DDIM": "ddpm", "VQDM": "vqdm", "Imagen": "other",
    "BigGAN": "gan", "DF-GAN": "gan", "GALIP": "gan", "GigaGAN": "gan",
    "starGAN": "gan", "styleGAN": "gan", "VQGAN": "gan",
    "MAE": "other", "MAGE": "other", "VQVAE": "other",
}

# PLAN C2 / D3: held out of training entirely, both tiers. Every independent
# identifier of the family is listed, so a renamed or re-rooted path cannot slip
# a DALL-E image through: the CSV file it came from, its Architecture/Category
# column, and the mapped generator_family all have to be clean.
PROTECTED_CSVS = ("dalle2.csv", "dalle3.csv")
PROTECTED_ARCHITECTURES = frozenset({"DALLE"})
PROTECTED_CATEGORIES = frozenset({"DALLE"})
PROTECTED_FAMILY = "dalle"
PROTECTED_PATH_TOKENS = ("/dalle", "dall-e", "dall_e")

Row = namedtuple("Row",
                 "csv_name key generator architecture category is_advanced is_fake")


def protected_reason(row: "Row") -> str:
    """Non-empty when this row belongs to the protected DALL-E family, by any
    identifier. Path is checked last precisely because it is the weakest."""
    if row.csv_name in PROTECTED_CSVS:
        return "wildfake_dalle"
    if row.architecture.upper() in PROTECTED_ARCHITECTURES:
        return "wildfake_dalle"
    if row.category.upper() in PROTECTED_CATEGORIES:
        return "wildfake_dalle"
    if family_of(row) == PROTECTED_FAMILY:
        return "wildfake_dalle"
    padded = f"/{row.key}"
    if any(tok in padded for tok in PROTECTED_PATH_TOKENS):
        return "wildfake_dalle"
    return ""


def _strip_images_prefix(p: str) -> str:
    low = p.lower()
    base = IMAGES_SUBDIR.lower()
    if low == base:           # an archive sitting at the Images root
        return ""
    return p[len(base) + 1:] if low.startswith(base + "/") else p


def wildfake_key(path: str) -> str:
    """Comparable key for a CSV `Image_path` or a manifest `path`: the location
    below WildFake/Images, forward slashes, lowercased."""
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if "#" in p:
        base, frag = p.split("#", 1)
        base_dir = base.rsplit("/", 1)[0] if "/" in base else ""
        rel = _strip_images_prefix(base_dir)
        p = f"{rel}/{frag}" if rel else frag
    else:
        p = _strip_images_prefix(p)
    return p.lower()


def csv_files(data_root) -> list[Path]:
    d = Path(data_root) / CSV_SUBDIR
    return sorted(d.glob("*.csv")) if d.exists() else []


def iter_csv_rows(data_root, only: list[str] | None = None):
    """Yield a Row per CSV line. `only` filters by CSV file name."""
    for f in csv_files(data_root):
        if only is not None and f.name not in only:
            continue
        with open(f, newline="", encoding="utf-8", errors="replace") as fh:
            for rec in csv.DictReader(fh):
                raw = rec.get("Image_path")
                if not raw:
                    continue
                yield Row(f.name, wildfake_key(raw), rec.get("Generator", ""),
                          rec.get("Architecture", ""), rec.get("Category", ""),
                          rec.get("IsAdvanced", "0") == "1",
                          rec.get("IsFake", "0") == "1")


def is_coco_val2017_key(key: str) -> bool:
    return "/coco2017/val2017/" in f"/{key}"


def coco_val2017_demo_ids(data_root) -> set[str]:
    """Basenames of the 4,998-image demonstration benchmark subset, as listed by
    WildFake's real_coco.csv. This is *not* the same as the 5,000 images in the
    canonical COCO val2017 archive - see build_denylist.py."""
    return {r.key.rsplit("/", 1)[-1]
            for r in iter_csv_rows(data_root, only=["real_coco.csv"])
            if is_coco_val2017_key(r.key)}


def protected_keys(data_root) -> dict[str, str]:
    """key -> reason for every path that must never be trainable (C2).

    * whole WildFake DALL-E family      -> "wildfake_dalle"
    * WildFake's own COCO val2017 copy  -> "coco_val2017"
    """
    out: dict[str, str] = {}
    for r in iter_csv_rows(data_root):
        reason = protected_reason(r)
        if reason:
            out[r.key] = reason
        elif is_coco_val2017_key(r.key):
            out[r.key] = "coco_val2017"
    return out


def protected_summary(data_root) -> dict:
    """Per-tier counts, so a builder can state what it actually excluded."""
    counts = {"wildfake_dalle_typical": 0, "wildfake_dalle_advanced": 0,
              "coco_val2017_demo": 0}
    for r in iter_csv_rows(data_root):
        if protected_reason(r):
            key = ("wildfake_dalle_advanced" if r.is_advanced
                   else "wildfake_dalle_typical")
            counts[key] += 1
        elif is_coco_val2017_key(r.key):
            counts["coco_val2017_demo"] += 1
    counts["wildfake_dalle_total"] = (counts["wildfake_dalle_typical"]
                                      + counts["wildfake_dalle_advanced"])
    return counts


def zip_prefix_map(data_root) -> dict[str, tuple[str, str]]:
    """csv-path prefix (lowercase) -> (manifest-relative zip path, zip top dir)."""
    images = Path(data_root) / IMAGES_SUBDIR
    out: dict[str, tuple[str, str]] = {}
    if not images.exists():
        return out
    for z in sorted(images.rglob("*.zip")):
        rel = z.relative_to(images).as_posix()
        out[rel[:-4].lower()] = (f"{IMAGES_SUBDIR}/{rel}", z.stem)
    return out


class ArchiveIndex:
    """Resolves a CSV key to a manifest path, skipping unavailable archives.

    Two layouts occur in this corpus:
      A. `Diffusion_based/DDIM.zip` holding `DDIM/imgs/x.png` — the archive name
         is a path component, resolved by prefix.
      B. `Diffusion_based/Midjourney/Typical/part_1.zip` holding `part_1/x.png`
         — a split archive whose name is *not* in the CSV path. Those are
         resolved by a member index built lazily, only on a prefix miss.

    Member lists are read once per archive; a partial download raises BadZipFile
    and the whole archive is marked unavailable.
    """

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.prefixes = zip_prefix_map(data_root)
        self.parents: dict[str, list[str]] = {}
        for prefix, (relzip, _) in self.prefixes.items():
            parent = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
            self.parents.setdefault(parent, []).append(relzip)
        self._members: dict[str, dict[str, str] | None] = {}
        self._suffixes: dict[str, dict[str, str]] = {}
        self.unavailable: set[str] = set()
        images = self.data_root / IMAGES_SUBDIR
        self._has_plain = images.exists() and any(
            p.is_file() and p.suffix.lower() != ".zip" for p in images.rglob("*"))

    def _member_map(self, relzip: str) -> dict[str, str] | None:
        """lowercase member name -> actual member name."""
        if relzip not in self._members:
            try:
                with zipfile.ZipFile(self.data_root / relzip) as zf:
                    self._members[relzip] = {n.lower(): n for n in zf.namelist()}
            except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
                print(f"[wildfake] archive unavailable {relzip}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
                self._members[relzip] = None
                self.unavailable.add(relzip)
        return self._members[relzip]

    def _suffix_map(self, relzip: str, parent: str) -> dict[str, str]:
        """Layout B: csv key -> member, dropping the archive's own top directory."""
        if relzip not in self._suffixes:
            members = self._member_map(relzip)
            out: dict[str, str] = {}
            for low, actual in (members or {}).items():
                if low.endswith("/") or "/" not in low:
                    continue
                tail = low.split("/", 1)[1]
                out[f"{parent}/{tail}" if parent else tail] = actual
            self._suffixes[relzip] = out
        return self._suffixes[relzip]

    def resolve(self, key: str) -> str | None:
        """key (lowercase, below WildFake/Images) -> manifest path, or None."""
        if self._has_plain and (self.data_root / IMAGES_SUBDIR / key).is_file():
            return f"{IMAGES_SUBDIR}/{key}"
        parts = key.split("/")
        for cut in range(len(parts) - 1, 0, -1):
            hit = self.prefixes.get("/".join(parts[:cut]))
            if hit is None:
                continue
            relzip, top = hit
            members = self._member_map(relzip)
            if members is None:
                return None
            actual = members.get("/".join([top] + parts[cut:]).lower())
            return f"{relzip}#{actual}" if actual else None
        # layout B: split archives sitting below a CSV directory
        for cut in range(len(parts) - 1, 0, -1):
            parent = "/".join(parts[:cut])
            for relzip in self.parents.get(parent, []):
                actual = self._suffix_map(relzip, parent).get(key)
                if actual:
                    return f"{relzip}#{actual}"
        return None


def family_of(row: Row) -> str:
    if not row.is_fake:
        return ""
    return ARCH_TO_FAMILY.get(row.architecture, "other")
