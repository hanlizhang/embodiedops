"""Simulation-only execution feedback. No door state or policy inputs here."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Rule:
    target_error_m: float = 0.01
    max_motion_m: float = 0.001
    loaded_normal_N: float = 1.0
    consecutive_steps: int = 5
    release_steps: int = 5
    cooldown_steps: int = 20
    max_events: int = 12


def contact_evidence(samples, robot_ids, fridge_ids):
    """Classify solver-loaded contacts by exact geometry membership."""
    peaks = dict(robot_fridge=0., robot_self=0., robot_environment=0.)
    pairs = {}
    for sample in samples:
        for c in sample['robot_contacts']:
            a, b = c['geom_ids']
            if c['constraint_address'] < 0:
                continue
            force = float(c['force_N_contact_frame'][0])
            if not np.isfinite(force):
                raise ValueError('Nonfinite contact force')
            if force <= 0:
                continue
            if a in robot_ids and b in robot_ids:
                kind = 'robot_self'
            elif (a in robot_ids and b in fridge_ids) or (b in robot_ids and a in fridge_ids):
                kind = 'robot_fridge'
            else:
                kind = 'robot_environment'
            peaks[kind] = max(peaks[kind], force)
            key = (kind, min(a, b), max(a, b))
            pairs[key] = max(pairs.get(key, 0.), force)
    return {'peak_normal_N': peaks, 'pairs': [dict(kind=k[0], geom_ids=list(k[1:]), peak_normal_N=v)
                                              for k, v in sorted(pairs.items())]}


def tracking_evidence(trace, action):
    """Use OSC's actual base-frame goal, including desired/achieved mode.

    Compare both world EEF positions in the SAME command-time controller frame.
    Target increment is not an expected one-step displacement.
    """
    t = trace['targets']['right']
    rotation, origin = np.asarray(t['origin_ori']), np.asarray(t['origin_pos'])
    before = rotation.T @ (np.asarray(trace['before']['robot0_eef_pos']) - origin)
    after = rotation.T @ (np.asarray(trace['after']['robot0_eef_pos']) - origin)
    target = np.asarray(t['goal_pos'])
    command, achieved = target - before, after - before
    values = np.concatenate([command, achieved, target-after])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite tracking evidence')
    return dict(command_target_delta_m=command.tolist(), achieved_delta_m=achieved.tolist(),
                target_error_m=float(np.linalg.norm(target-after)),
                achieved_motion_m=float(np.linalg.norm(achieved)),
                scaled_public_translation_m=(np.clip(action[5:8], -1, 1)*.05).tolist(),
                goal_update_mode=t['_goal_update_mode'], target_base_m=target.tolist(),
                origin_world_m=origin.tolist(), origin_rotation=rotation.tolist())


def valid_cameras(observation):
    return all(np.isfinite(v).all() and np.max(v) > 0 for v in observation['pixels'].values())


class Supervisor:
    def __init__(self, rule):
        self.rule = rule
        self.streak = self.release = self.events = 0
        self.armed = True
        self.last_event = -rule.cooldown_steps

    def update(self, step, tracking, contacts, remaining, safe=True):
        r = self.rule
        loaded = contacts['peak_normal_N']['robot_fridge'] >= r.loaded_normal_N
        blocked = (loaded and tracking['target_error_m'] >= r.target_error_m
                   and tracking['achieved_motion_m'] <= r.max_motion_m)
        self.streak = self.streak + 1 if blocked and safe else 0
        self.release = 0 if loaded else self.release + 1
        if self.release >= r.release_steps and step-self.last_event >= r.cooldown_steps:
            self.armed = True
        fire = (safe and self.armed and self.streak >= r.consecutive_steps and remaining > 0
                and step-self.last_event >= r.cooldown_steps and self.events < r.max_events)
        if fire:
            self.events += 1
            self.last_event = step
            self.armed = False
            self.streak = 0
        return dict(trigger=bool(fire), reason='loaded_fridge_tracking_stall' if fire else None,
                    step=step, armed=self.armed, blocked=bool(blocked), streak=self.streak,
                    events=self.events, actions_discarded=remaining if fire else 0)


def interrupt(policy, decision):
    """Leave controller and native refill semantics intact; never reset the policy."""
    if decision['trigger']:
        assert len(policy._queues['action']) == decision['actions_discarded']
        policy._queues['action'].clear()
