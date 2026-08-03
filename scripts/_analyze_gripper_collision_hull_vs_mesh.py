"""Pi-local (no Isaac Sim/GPU) check of AR4 gripper COLLISION-hull vs VISUAL-mesh
size mismatch — prompted by a direct user observation (2026-08-03) that the
collision bounding box and the cube/gripper mesh in the success video are not
the same size.

Downloads the vendor gripper STLs (ycheng517/ar4_ros_driver
annin_ar4_description/meshes/ar4_gripper/{gripper_base_link,gripper_jaw1_link,
gripper_jaw2_link}.stl) and compares each link's raw triangle-mesh solid volume
to its convex-hull volume (the approximation PhysX actually uses, per
UsdPhysics.MeshCollisionAPI approximation="convexHull" authored on all three
links, confirmed 2026-07-21).

RESULT (2026-08-03): hull/mesh volume ratios jaw1=2.38x, jaw2=2.30x,
base=1.54x. The gripper fingers have a concave gripping notch that the convex
hull fills solid, so the COLLISION gripper is ~2.3x fatter than the real
fingers with a bulged inner face / narrower effective aperture. This confirms
the collision geometry is substantially inflated vs the visual mesh --
evidence the "gripper too big to straddle" conclusion is at least partly a
convex-hull approximation artifact, fixable via convexDecomposition or mesh
(sdf/triangle) collision that preserves the notch/gap.

Run: python3 -m venv .venv && .venv/bin/pip install numpy scipy
     (download the 3 STLs next to this script first), then .venv/bin/python this.
"""

import struct, numpy as np
from scipy.spatial import ConvexHull

def load_stl(path):
    with open(path,'rb') as f:
        data=f.read()
    # detect ascii
    if data[:5].lower()==b'solid' and b'facet' in data[:512].lower():
        verts=[]
        for line in data.decode('ascii','ignore').splitlines():
            s=line.strip().split()
            if len(s)>=4 and s[0]=='vertex':
                verts.append([float(s[1]),float(s[2]),float(s[3])])
        return np.array(verts).reshape(-1,3)
    # binary
    n=struct.unpack('<I',data[80:84])[0]
    tris=np.frombuffer(data[84:84+n*50],dtype=np.uint8).reshape(n,50)
    v=np.zeros((n,3,3))
    for i in range(n):
        vals=struct.unpack('<12f',tris[i,:48].tobytes())
        v[i]=np.array(vals[3:12]).reshape(3,3)
    return v.reshape(-1,3)

def tri_volume(path):
    # signed volume of a closed triangle mesh (STL triangles)
    with open(path,'rb') as f: data=f.read()
    n=struct.unpack('<I',data[80:84])[0]
    vol=0.0
    off=84
    for i in range(n):
        vals=struct.unpack('<12f',data[off:off+48]); off+=50
        a=np.array(vals[3:6]); b=np.array(vals[6:9]); c=np.array(vals[9:12])
        vol+=np.dot(a,np.cross(b,c))/6.0
    return abs(vol)

for name in ['gripper_base_link','gripper_jaw1_link','gripper_jaw2_link']:
    p=f'{name}.stl'
    pts=load_stl(p)
    mn=pts.min(0); mx=pts.max(0); ext=mx-mn
    hull=ConvexHull(pts)
    aabb_vol=np.prod(ext)
    mesh_vol=tri_volume(p)
    print(f'\n=== {name} ===')
    print(f'  raw mesh AABB extent (mm): X={ext[0]*1000:.1f} Y={ext[1]*1000:.1f} Z={ext[2]*1000:.1f}')
    print(f'  AABB min (mm): {np.round(mn*1000,1)}  max: {np.round(mx*1000,1)}')
    print(f'  mesh solid volume : {mesh_vol*1e9:8.1f} mm^3')
    print(f'  convex hull volume: {hull.volume*1e9:8.1f} mm^3')
    print(f'  hull/mesh ratio   : {hull.volume/mesh_vol:6.2f}x   (>>1 = hull fills big concavity)')
    print(f'  hull/AABB fill    : {hull.volume/aabb_vol*100:5.1f}%   (how much of the bounding box the hull solidly fills)')
