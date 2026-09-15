#!/usr/bin/env python3
"""Convert Pollen Robotics MicroDuck MJCF (robot_walk.xml) into robot.json
for the Three.js kinematic rebuild. Body quats are MuJoCo (w,x,y,z)."""
import json, struct, sys, xml.etree.ElementTree as ET

MJCF = sys.argv[1] if len(sys.argv) > 1 else 'robot_walk.xml'
ASSET_DIR = '../assets/meshes'
root = ET.parse(MJCF).getroot()
world = root.find('worldbody')

def float3(s, default=(0,0,0)):
    if s is None: return list(default)
    return [float(x) for x in s.split()]
def quat4(s):
    if s is None: return [1,0,0,0]  # w x y z
    return [float(x) for x in s.split()]

def stl_bbox(path):
    try:
        with open(path,'rb') as f:
            f.read(80)
            n = struct.unpack('<I', f.read(4))[0]
            lo=[1e9]*3; hi=[-1e9]*3
            for _ in range(n):
                f.read(12)
                for _ in range(3):
                    v = struct.unpack('<3f', f.read(12))
                    for k in range(3):
                        lo[k]=min(lo[k],v[k]); hi[k]=max(hi[k],v[k])
                f.read(2)
            return [hi[k]-lo[k] for k in range(3)]
    except OSError:
        return None

bodies = []
def walk(el, parent):
    for b in el.findall('body'):
        name = b.get('name')
        joint = None
        j = b.find('joint')
        if j is not None:
            joint = {'name': j.get('name'), 'axis': float3(j.get('axis'), (0,0,1)),
                     'range': [float(x) for x in j.get('range').split()]}
        meshes = []
        for g in b.findall('geom'):
            mesh = g.get('mesh')
            if mesh:
                meshes.append({'file': mesh + '.stl', 'pos': float3(g.get('pos')),
                               'quat': quat4(g.get('quat'))})
        bodies.append({'name': name, 'parent': parent,
                       'pos': float3(b.get('pos')), 'quat': quat4(b.get('quat')),
                       'joint': joint, 'meshes': meshes})
        walk(b, name)

walk(world, None)

# unit sanity: xl330 servo is ~23.5mm across its largest dimension
bb = stl_bbox(f'{ASSET_DIR}/xl330.stl')
scale = 0.001 if bb and max(bb) > 0.5 else 1.0
print('xl330 bbox:', [round(x,4) for x in bb], '-> mesh scale', scale)

out = {'mesh_scale': scale, 'bodies': bodies}
with open('../assets/robot.json','w') as f:
    json.dump(out, f, indent=1)
nj = sum(1 for b in bodies if b['joint'])
nm = sum(len(b['meshes']) for b in bodies)
print(f'bodies={len(bodies)} joints={nj} geom-meshes={nm}')
print('joints:', [b['joint']['name'] for b in bodies if b['joint']])
