"""Verify the rebuilt composed AR4 asset: (1) every visual mesh's bound material
now carries a non-white diffuse matching the URDF scheme, and (2) collision
geometry is still present on all links. Instance-proxy-aware traversal."""

import os
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(REPO_ROOT, "assets", "ar4_mk5", "ar4_mk5.usd")

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, UsdShade

stage = Usd.Stage.Open(USD)

visual_meshes = 0
collision_meshes = 0
white_visuals = []
color_hist = Counter()
collision_links = set()
per_link = {}

for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
    path = prim.GetPath().pathString
    tname = prim.GetTypeName()
    if tname == "Mesh":
        is_collision = prim.HasAPI(UsdPhysics.CollisionAPI) or "/collisions/" in path
        if is_collision:
            collision_meshes += 1
            # link name = component after root_joint
            parts = path.split("/")
            if "root_joint" in parts:
                i = parts.index("root_joint")
                if i + 1 < len(parts):
                    collision_links.add(parts[i + 1])
        else:
            visual_meshes += 1
            binding = UsdShade.MaterialBindingAPI(prim)
            mat, _ = binding.ComputeBoundMaterial()
            diffuse = None
            if mat and mat.GetPrim().IsValid():
                # find surface shader
                for child in Usd.PrimRange(mat.GetPrim()):
                    if child.GetTypeName() == "Shader":
                        sh = UsdShade.Shader(child)
                        inp = sh.GetInput("diffuse_color_constant")
                        if inp and inp.Get() is not None:
                            diffuse = tuple(round(float(c), 3) for c in inp.Get())
                            break
            color_hist[diffuse] += 1
            if diffuse == (1.0, 1.0, 1.0) or diffuse is None:
                white_visuals.append(path)
            # link grouping
            parts = path.split("/")
            if "root_joint" in parts:
                i = parts.index("root_joint")
                link = parts[i + 1] if i + 1 < len(parts) else "?"
                per_link.setdefault(link, Counter())[diffuse] += 1

print("\n========== COLOR VERIFICATION ==========")
print(f"visual meshes:    {visual_meshes}")
print(f"collision meshes: {collision_meshes}")
print(f"links with collision geometry: {sorted(collision_links)}")
print(f"\ndiffuse color histogram (color -> #visual meshes):")
for color, n in sorted(color_hist.items(), key=lambda kv: (-kv[1])):
    print(f"    {color} : {n}")
print(f"\nvisual meshes still WHITE/none: {len(white_visuals)}")
for w in white_visuals:
    print(f"    {w}")
print("\nper-link visual color breakdown:")
for link in sorted(per_link):
    print(f"  {link}: {dict(per_link[link])}")

app.close()
