#!/usr/bin/env python3
"""Apply a narrow, reversible late-capture scoring fix to src/sim/env.py.

Usage (from a repository checkout):
    python /path/to/apply_boundary_fix.py src/sim/env.py

Designed for the source blocks published in commit
2a4f1df6449443fd6f0ade8d97b11fb65876bf7c. Refuses incompatible blocks.
Creates an adjacent .before_boundary_fix backup. Never changes heuristics,
arrival generation, time step, guidance, or decision epochs.

Scope: only original endpoint-detected captures at x<=0 are reconsidered.
Continuous missed contacts and immediate substep reselection remain out of scope.
Motion within a step is interpreted as moving at v_interceptor toward the
original guidance destination, then remaining there if it is reached early.
Strict temporal precedence is required; times within 1e-10 are treated as tied.
"""
from __future__ import annotations
import argparse
from pathlib import Path

MARKER = '# BEGIN NARROW LATE-CAPTURE SCORING FIX'
HELPER = r'''
# BEGIN NARROW LATE-CAPTURE SCORING FIX

def _boundary_fix_segment_contact(r, w, radius, duration):
    """First contact in a constant-relative-velocity segment, or infinity."""
    c = float(np.dot(r, r) - radius * radius)
    if c <= 0.0:
        return 0.0
    a = float(np.dot(w, w))
    if a == 0.0:
        return np.inf
    b = float(2.0 * np.dot(r, w))
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return np.inf
    # Stable roots of a*t^2+b*t+c=0.
    q = -0.5 * (b + np.copysign(np.sqrt(discriminant), b))
    roots = [q / a, c / q] if q != 0.0 else [-b / (2.0 * a)]
    for value in sorted(roots):
        if 0.0 <= value <= duration + 1e-12:
            return min(float(value), duration)
    return np.inf


def _boundary_fix_contact_time(pI0, destination, pT0, vT, speed, radius, dt):
    """Respect move_toward's speed cap and possible midstep stop."""
    delta = destination - pI0
    distance = float(np.linalg.norm(delta))
    if distance < 1e-12 or speed <= 0.0:
        return _boundary_fix_segment_contact(pT0 - pI0, vT, radius, dt)
    moving_duration = min(dt, distance / speed)
    interceptor_velocity = delta * (speed / distance)
    hit = _boundary_fix_segment_contact(
        pT0 - pI0, vT - interceptor_velocity, radius, moving_duration
    )
    if np.isfinite(hit):
        return hit
    if moving_duration < dt:
        relative_start = pT0 + vT * moving_duration - destination
        hit = _boundary_fix_segment_contact(
            relative_start, vT, radius, dt - moving_duration
        )
        if np.isfinite(hit):
            return moving_duration + hit
    return np.inf


def _boundary_fix_valid_capture(pI0, destination, pT0, vT, speed, radius, dt):
    """A target must first enter the radius strictly BEFORE reaching x=0."""
    if pT0[0] <= 0.0:
        return False
    crossing_time = -float(pT0[0]) / float(vT[0]) if vT[0] < 0.0 else np.inf
    hit_time = _boundary_fix_contact_time(
        pI0, destination, pT0, vT, speed, radius, dt
    )
    return bool(np.isfinite(hit_time) and hit_time < crossing_time - 1e-10)

# END NARROW LATE-CAPTURE SCORING FIX
'''
OLD_GUIDANCE = '''        # 2. Lead-pursuit interceptor guidance based on the current state.
        target = self._select_target_object(target_id)
'''
NEW_GUIDANCE = '''        # 2. Lead-pursuit interceptor guidance based on the current state.
        # Save positions AFTER births, BEFORE movement; RNG calls are unchanged.
        boundary_fix_start_I = self.interceptor_pos.copy()
        boundary_fix_start_T = {
            th.id: th.pos.copy() for th in self.active_threats()
        }
        target = self._select_target_object(target_id)
'''
OLD_CAPTURE = '''            if float(np.linalg.norm(th.pos - self.interceptor_pos)) <= self.p.kill_radius:
                victims.append(th)
'''
NEW_CAPTURE = '''            if float(np.linalg.norm(th.pos - self.interceptor_pos)) <= self.p.kill_radius:
                if th.pos[0] <= 0.0 and not _boundary_fix_valid_capture(
                    boundary_fix_start_I,
                    destination,
                    boundary_fix_start_T[th.id],
                    th.vel,
                    self.p.v_interceptor,
                    self.p.kill_radius,
                    self.p.dt,
                ):
                    # Leave the target active for the existing escape loop below.
                    # Terminal time and subsequent decision timing stay unchanged.
                    self.rejected_late_intercepts = (
                        getattr(self, "rejected_late_intercepts", 0) + 1
                    )
                    continue
                victims.append(th)
'''

def patched_text(source: str) -> str:
    if MARKER in source:
        raise ValueError('This file already contains the late-capture fix.')
    replacements = [
        ('import numpy as np\n', 'import numpy as np\n' + HELPER + '\n'),
        (OLD_GUIDANCE, NEW_GUIDANCE),
        (OLD_CAPTURE, NEW_CAPTURE),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Expected exactly one compatible source block; aborting without changes.\n' + old)
        source = source.replace(old, new, 1)
    compile(source, '<patched env.py>', 'exec')
    return source

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('env_file', type=Path)
    args = parser.parse_args()
    path = args.env_file.resolve()
    if not path.is_file():
        parser.error(f'Not a file: {path}')
    original = path.read_text(encoding='utf-8')
    try:
        changed = patched_text(original)
    except ValueError as exc:
        parser.error(str(exc))
    backup = path.with_name(path.name + '.before_boundary_fix')
    if backup.exists():
        parser.error(f'Backup already exists; refusing to overwrite: {backup}')
    backup.write_bytes(path.read_bytes())
    temporary = path.with_name(path.name + '.boundary_fix_tmp')
    temporary.write_text(changed, encoding='utf-8')
    temporary.replace(path)
    print(f'Patched: {path}\nBackup: {backup}')
    print('Run unit and regression tests before replacing any published results.')

if __name__ == '__main__':
    main()
