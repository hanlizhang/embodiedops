"""Static MuJoCo distance-API check: no integration, robot, rendering or policy."""
import json
import mujoco
import numpy as np

XML='''<mujoco><asset><mesh name="cube" vertex="-1 -1 -1 1 -1 -1 -1 1 -1 1 1 -1 -1 -1 1 1 -1 1 -1 1 1 1 1 1"/></asset>
<worldbody><geom name="hand" type="mesh" mesh="cube"/><geom name="door" type="box" size="1 1 1" pos="4 0 0"/></worldbody></mujoco>'''


def ordered_geometry_distance(model,data,first,second,cutoff):
    """Canonical call order avoids 3.3.1 native-CCD early-return endpoint swap.

    Query in ascending geometry-type order, then explicitly restore caller order.
    Works for both native CCD and the primitive/libccd fallback without toggling
    any simulator flag. Endpoint interpretation is undefined at the cutoff.
    """
    swapped=model.geom_type[first]>model.geom_type[second]
    a,b=(second,first) if swapped else (first,second)
    points=np.zeros(6)
    distance=mujoco.mj_geomDistance(model,data,a,b,cutoff,points)
    if swapped: points=np.concatenate((points[3:],points[:3]))
    return float(distance),points


def audit():
    m=mujoco.MjModel.from_xml_string(XML); d=mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    result=dict(version=mujoco.__version__,integration_steps=0,queries=[])
    for a,b in [(0,1),(1,0)]:
        points=np.zeros(6)
        distance=mujoco.mj_geomDistance(m,d,a,b,10.,points)
        corrected_distance,corrected=ordered_geometry_distance(m,d,a,b,10.)
        result['queries'].append(dict(first=m.geom(a).name,second=m.geom(b).name,
            distance=float(distance),points=points.tolist(),
            expected_direction_x=1 if a==0 else -1,
            returned_direction_x=float(np.sign(points[3]-points[0])),
            corrected_points=corrected.tolist(),corrected_distance=corrected_distance,
            corrected_direction_x=float(np.sign(corrected[3]-corrected[0]))))
    return result


if __name__=='__main__': print(json.dumps(audit(),indent=2))
