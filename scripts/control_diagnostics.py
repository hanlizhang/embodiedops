"""Read-only physics diagnostics; no observations are added to policy inputs."""
import json
import time
from contextlib import contextmanager

import numpy as np


def contact_snapshot(sim):
    """Solver contacts, not geometric proximity. Wrench uses contact-local axes."""
    import mujoco
    model, data = sim.model._model, sim.data._data
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        assert np.isfinite(wrench).all()
        rows.append({"geom_ids": [int(c.geom1), int(c.geom2)],
                     "distance_m": float(c.dist), "position_m": c.pos.tolist(),
                     "constraint_address": int(c.efc_address),
                     "force_N_contact_frame": wrench[:3].tolist(),
                     "torque_Nm_contact_frame": wrench[3:].tolist()})
    return rows


def geometry_catalog(sim):
    import mujoco
    m = sim.model._model
    result = []
    for gid in range(m.ngeom):
        bid = int(m.geom_bodyid[gid])
        ancestors = []
        while bid:
            ancestors.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid))
            bid = int(m.body_parentid[bid])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or str(gid)
        result.append({"id": gid, "name": name, "body_ancestors": ancestors,
                       "robot": any(s.startswith(("robot0_", "gripper0_")) for s in [name, *ancestors])})
    return result


class ContactRecorder:
    def __init__(self, sim, output):
        self.sim = sim
        self.catalog = geometry_catalog(sim)
        (output / 'geometry-catalog.json').write_text(json.dumps(self.catalog, indent=2))
        self.robot_ids = {g['id'] for g in self.catalog if g['robot']}
        assert self.robot_ids
        self.output = output

    @contextmanager
    def step(self, index):
        sim = self.sim
        samples = []
        original = {name: getattr(sim, name) for name in ('step', 'step2')}
        def wrap(fn):
            def measured(*args, **kwargs):
                result = fn(*args, **kwargs)
                contacts = contact_snapshot(sim)
                samples.append({'time': float(sim.data.time), 'ncon': len(contacts),
                                'robot_contacts': [c for c in contacts if self.robot_ids.intersection(c['geom_ids'])]})
                return result
            return measured
        for name, fn in original.items():
            setattr(sim, name, wrap(fn))
        start = time.perf_counter()
        try:
            yield
        finally:
            self.last_step_seconds = time.perf_counter() - start
            for name, fn in original.items():
                setattr(sim, name, fn)
            with (self.output / 'contacts.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'step': index, 'samples': samples}) + '\n')
            self.last_sample_count = len(samples)
            self.last_samples = samples


def action_change(previous, current):
    if previous is None:
        return None
    d = np.asarray(current) - np.asarray(previous)
    continuous = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    return {'per_dimension': d.tolist(), 'continuous_l2': float(np.linalg.norm(d[continuous])),
            'mode_switch': bool((previous[4] >= .5) != (current[4] >= .5)),
            'gripper_switch': bool((previous[11] >= .5) != (current[11] >= .5))}
