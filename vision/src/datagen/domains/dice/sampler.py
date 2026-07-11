import random
from dataclasses import dataclass
from typing import Optional

DIE_TYPES = ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]

SIZE_RANGES_MM = {
    "d4": (14.0, 20.0),
    "d6": (12.0, 20.0),
    "d8": (14.0, 20.0),
    "d10": (14.0, 20.0),
    "d10_pct": (14.0, 20.0),
    "d12": (16.0, 22.0),
    "d20": (16.0, 24.0),
}

MATERIAL_CATEGORIES = ["opaque", "translucent", "marbled", "glitter", "metallic", "speckled"]
GLYPH_STYLES = ["arabic_numerals", "roman_numerals", "pips", "greek_numerals", "cjk_numerals"]
GLYPH_METHODS = ["engraved", "printed_decal"]
GLYPH_FILLS = ["painted", "blank"]
D4_PLACEMENT_STYLES = ["face_centered", "vertex_labeled"]
FONT_IDS = ["font_sans_bold", "font_serif_regular", "font_display_condensed"]


@dataclass
class DiceVariantParams:
    die_type: str
    size_mm: float
    bevel_fraction: float
    numbering_scheme: str
    glyph_style: str
    glyph_method: str
    glyph_fill: str
    font_or_style_id: str
    material_category: str
    material_params: dict
    d4_placement: Optional[str]
    seed: int


def sample_variant(seed: int) -> DiceVariantParams:
    rng = random.Random(seed)

    die_type = rng.choice(DIE_TYPES)
    lo, hi = SIZE_RANGES_MM[die_type]
    size_mm = rng.uniform(lo, hi)
    bevel_fraction = rng.uniform(0.02, 0.06)

    if die_type in ("d6", "d4"):
        glyph_style = rng.choice(["arabic_numerals", "pips"])
    elif die_type == "d10_pct":
        glyph_style = "arabic_numerals"
    elif die_type == "d10":
        # d10's value range includes 0, and neither the Roman nor the
        # (Milesian) Greek numeral system has a zero -- both previously
        # fell back to a lone arabic "0" face amid otherwise Roman/Greek
        # lettering, i.e. visibly mismatched. Only styles whose numeral
        # system actually covers 0 are allowed here.
        glyph_style = rng.choice(["arabic_numerals", "cjk_numerals"])
    else:
        glyph_style = rng.choice([s for s in GLYPH_STYLES if s != "pips"])

    glyph_method = rng.choice(GLYPH_METHODS)
    # Engraved fill is always painted: an unpainted (blank) engraving at
    # the current shallow ENGRAVE_DEPTH_FRACTION is nearly invisible, and
    # readable numerals are a hard requirement for training data.
    glyph_fill = "painted"
    font_or_style_id = rng.choice(FONT_IDS)

    material_category = rng.choice(MATERIAL_CATEGORIES)
    material_params = _sample_material_params(rng, material_category)

    d4_placement = rng.choice(D4_PLACEMENT_STYLES) if die_type == "d4" else None

    return DiceVariantParams(
        die_type=die_type,
        size_mm=size_mm,
        bevel_fraction=bevel_fraction,
        numbering_scheme=f"standard_{die_type}",
        glyph_style=glyph_style,
        glyph_method=glyph_method,
        glyph_fill=glyph_fill,
        font_or_style_id=font_or_style_id,
        material_category=material_category,
        material_params=material_params,
        d4_placement=d4_placement,
        seed=seed,
    )


def sample_set(seed: int) -> dict:
    rng = random.Random(seed)

    material_category = rng.choice(MATERIAL_CATEGORIES)
    material_params = _sample_material_params(rng, material_category)
    glyph_method = rng.choice(GLYPH_METHODS)
    # Always painted -- see sample_variant's identical rule.
    glyph_fill = "painted"
    font_or_style_id = rng.choice(FONT_IDS)
    bevel_fraction = rng.uniform(0.02, 0.06)
    glyph_style = rng.choice([s for s in GLYPH_STYLES if s != "pips"])

    variants = {}
    for die_type in DIE_TYPES:
        lo, hi = SIZE_RANGES_MM[die_type]
        size_mm = rng.uniform(lo, hi)
        d4_placement = rng.choice(D4_PLACEMENT_STYLES) if die_type == "d4" else None
        if die_type == "d10_pct":
            die_glyph_style = "arabic_numerals"
        elif die_type == "d10" and glyph_style in ("roman_numerals", "greek_numerals"):
            # Roman and Greek numeral systems have no zero (see
            # sample_variant) -- the d10 in a Roman/Greek-lettered set
            # falls back to arabic, mirroring d10_pct's override, rather
            # than showing one mismatched arabic "0" face.
            die_glyph_style = "arabic_numerals"
        else:
            die_glyph_style = glyph_style

        variants[die_type] = DiceVariantParams(
            die_type=die_type,
            size_mm=size_mm,
            bevel_fraction=bevel_fraction,
            numbering_scheme=f"standard_{die_type}",
            glyph_style=die_glyph_style,
            glyph_method=glyph_method,
            glyph_fill=glyph_fill,
            font_or_style_id=font_or_style_id,
            material_category=material_category,
            material_params=material_params,
            d4_placement=d4_placement,
            seed=seed,
        )

    return variants


def _sample_material_params(rng, category):
    params = {
        "hue": rng.uniform(0.0, 1.0),
        "saturation": rng.uniform(0.3, 1.0),
        "value": rng.uniform(0.2, 0.9),
        "roughness": rng.uniform(0.1, 0.7),
    }
    if category == "translucent":
        params["ior"] = rng.uniform(1.3, 1.6)
        params["transmission"] = rng.uniform(0.7, 1.0)
    elif category == "marbled":
        params["noise_scale"] = rng.uniform(2.0, 8.0)
        params["secondary_hue"] = rng.uniform(0.0, 1.0)
    elif category == "glitter":
        params["sparkle_density"] = rng.uniform(20.0, 80.0)
    elif category == "metallic":
        params["roughness"] = rng.uniform(0.05, 0.35)
    elif category == "speckled":
        params["speckle_density"] = rng.uniform(30.0, 100.0)
        params["secondary_hue"] = rng.uniform(0.0, 1.0)
    return params
