"""
Procedural shader-node materials for the 6 realistic dice finish categories.
Node input/output socket names below (e.g. "Transmission Weight", "Factor")
were confirmed against this project's installed Blender 5.1.2 — Blender has
renamed several Principled BSDF and texture-node sockets across versions
(e.g. "Transmission" -> "Transmission Weight", noise/ramp "Fac" -> "Factor").
"""
import colorsys

import bpy

MATERIAL_CATEGORIES = ["opaque", "translucent", "marbled", "glitter", "metallic", "speckled"]


def _hsv_to_rgba(h, s, v, a=1.0):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, a)


def build_material(die_name, category, params):
    mat = bpy.data.materials.new(name=f"{die_name}_{category}")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    base_color = _hsv_to_rgba(params["hue"], params["saturation"], params["value"])
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = params["roughness"]
    mat.diffuse_color = base_color

    if category == "opaque":
        pass

    elif category == "translucent":
        bsdf.inputs["Transmission Weight"].default_value = params.get("transmission", 0.9)
        bsdf.inputs["IOR"].default_value = params.get("ior", 1.45)

    elif category == "marbled":
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = params.get("noise_scale", 5.0)
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        secondary = _hsv_to_rgba(params.get("secondary_hue", 0.0), params["saturation"], params["value"])
        ramp.color_ramp.elements[0].color = base_color
        ramp.color_ramp.elements[1].color = secondary
        nt.links.new(noise.outputs["Factor"], ramp.inputs["Factor"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    elif category == "glitter":
        voronoi = nt.nodes.new("ShaderNodeTexVoronoi")
        voronoi.inputs["Scale"].default_value = params.get("sparkle_density", 40.0)
        voronoi.feature = 'DISTANCE_TO_EDGE'
        bsdf.inputs["Metallic"].default_value = 0.6
        nt.links.new(voronoi.outputs["Distance"], bsdf.inputs["Roughness"])

    elif category == "metallic":
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = params.get("roughness", 0.15)

    elif category == "speckled":
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = params.get("speckle_density", 60.0)
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.45
        ramp.color_ramp.elements[1].position = 0.55
        secondary = _hsv_to_rgba(params.get("secondary_hue", 0.0), params["saturation"], params["value"])
        ramp.color_ramp.elements[0].color = base_color
        ramp.color_ramp.elements[1].color = secondary
        nt.links.new(noise.outputs["Factor"], ramp.inputs["Factor"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    else:
        raise ValueError(f"unknown material category: {category!r}")

    return mat


def apply_material(die_obj, mat, slot_index=0):
    materials = die_obj.data.materials
    while len(materials) <= slot_index:
        materials.append(None)
    materials[slot_index] = mat


def build_fill_material(die_name, params, base_luminance=None):
    """
    Plain-color material for painted glyph fill (material slot 1).

    Contrast comes from LIGHTNESS opposition, not hue opposition: the
    previous complementary-hue fill at fixed s=0.8/v=0.9 could land at
    nearly the same luminance as the base (e.g. a bright die with a
    bright complementary fill), leaving engraved numerals hard to read.
    Real dice paint numerals in near-white or near-black ink; mirror
    that -- a dark base gets a light fill, a light base gets a dark
    fill, with the complementary hue kept only as a subtle tint.

    base_luminance: the base material's REAL rendered luminance (see
    glyphs.material_rendered_luminance). Prefer it whenever available:
    params["value"] is only a node-graph input and can badly misstate
    the rendered appearance (a translucent value=0.55 die rendered dark
    olive, making the param-chosen dark fill invisible). The param is
    kept only as a fallback for callers without a rendered swatch.
    """
    fill_hue = (params["hue"] + 0.5) % 1.0
    base_is_light = (
        base_luminance >= 0.5 if base_luminance is not None
        else params["value"] >= 0.5
    )
    if base_is_light:
        fill_color = _hsv_to_rgba(fill_hue, 0.25, 0.05)
    else:
        fill_color = _hsv_to_rgba(fill_hue, 0.12, 0.95)
    mat = bpy.data.materials.new(name=f"{die_name}_fill")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = fill_color
    bsdf.inputs["Roughness"].default_value = 0.4
    mat.diffuse_color = fill_color
    return mat
