# scripts/_diag_dice_material_check.py
"""One-off diagnostic (not a Gate A/P/G component): opens each dice_sets_v1
set_00000 die USD directly (no scene spawn, no physics) and prints every
Material/Shader prim found on the stage - inputs, connections, and which
mesh prims (if any) have a UsdShade.MaterialBindingAPI binding pointing at
one - to determine WHY these dice render near-white in Isaac despite being
authored with colored procedural materials in Blender (per each die's own
set_00000_<type>.json manifest material_category/material_params).

Same headless-SimulationApp asset-inspection pattern as
scripts/_diag_die_scale_check.py (plain python3/python.sh cannot import pxr
outside the full Kit environment - confirmed in that script's own history,
see .superpowers/sdd/dice-demo-report.md).

.. code-block:: bash

    flock /tmp/rl_isaac_sim.lock -c "/home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_dice_material_check.py"
"""

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdShade  # noqa: E402

DIE_TYPES = ["d4", "d8", "d10", "d12", "d20"]
DICE_SET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vision", "data", "raw", "dice_sets_v1"
)


def _describe_shader(prim: Usd.Prim) -> None:
    shader = UsdShade.Shader(prim)
    shader_id = shader.GetIdAttr().Get()
    print(f"    Shader prim: {prim.GetPath()} id={shader_id}")
    for inp in shader.GetInputs():
        val = None
        try:
            val = inp.Get()
        except Exception as e:  # noqa: BLE001
            val = f"<error reading: {e}>"
        conn = inp.GetConnectedSources()
        conn_str = ""
        if conn and conn[0]:
            conn_str = f" <- connected to {[str(c.source.GetPath()) for c in conn[0]]}"
        print(f"      input:{inp.GetBaseName()} = {val}{conn_str}")


def main() -> None:
    for die_type in DIE_TYPES:
        usd_path = os.path.join(DICE_SET_DIR, f"set_00000_{die_type}.usd")
        stage = Usd.Stage.Open(usd_path)
        print(f"\n===== {die_type} ({usd_path}) =====")

        material_prims = []
        shader_prims = []
        mesh_prims = []
        for prim in stage.Traverse():
            if prim.IsA(UsdShade.Material):
                material_prims.append(prim)
            if prim.IsA(UsdShade.Shader):
                shader_prims.append(prim)
            if prim.IsA(UsdGeom.Mesh):
                mesh_prims.append(prim)

        print(f"  mesh_count={len(mesh_prims)} material_count={len(material_prims)} shader_count={len(shader_prims)}")

        for mat_prim in material_prims:
            print(f"  Material prim: {mat_prim.GetPath()}")
            material = UsdShade.Material(mat_prim)
            surface_output = material.GetSurfaceOutput()
            if surface_output and surface_output.HasConnectedSource():
                src = surface_output.GetConnectedSources()[0][0]
                print(f"    surface output connected to: {src.source.GetPath()}")
            else:
                print("    surface output: NOT connected")

        for shader_prim in shader_prims:
            _describe_shader(shader_prim)

        print("  Mesh material bindings:")
        for mesh_prim in mesh_prims:
            binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
            bound = binding_api.ComputeBoundMaterial()
            mat = bound[0] if bound else None
            if mat and mat.GetPrim().IsValid():
                print(f"    {mesh_prim.GetPath()} -> BOUND to {mat.GetPath()}")
            else:
                print(f"    {mesh_prim.GetPath()} -> NO material binding")

            # Also check for any displayColor primvar (a common non-material
            # fallback USD uses for basic per-mesh color when no material is
            # bound, or when a viewer doesn't resolve the material).
            mesh = UsdGeom.Mesh(mesh_prim)
            display_color_attr = mesh.GetDisplayColorAttr()
            if display_color_attr and display_color_attr.HasAuthoredValue():
                print(f"    {mesh_prim.GetPath()} displayColor={display_color_attr.Get()}")
            else:
                print(f"    {mesh_prim.GetPath()} displayColor=<not authored>")


if __name__ == "__main__":
    main()
    simulation_app.close()
