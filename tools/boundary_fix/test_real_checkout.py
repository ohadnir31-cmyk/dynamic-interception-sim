"""Run analytic boundary cases against the user's real checkout (not audit_core)."""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--version', choices=['original', 'fixed'], required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    env_module = importlib.import_module('src.sim.env')
    actual = Path(env_module.__file__).resolve()
    expected = repo / 'src/sim/env.py'
    if actual != expected:
        raise RuntimeError(f'Wrong module imported: {actual}; expected {expected}')
    fixed = args.version == 'fixed'
    has_fix = hasattr(env_module, '_boundary_fix_valid_capture')
    if has_fix != fixed:
        raise RuntimeError('The imported simulator does not match --version.')
    print(f'Imported simulator: {actual}', flush=True)
    for title, start_x, fixed_success in [
        ('contact AFTER crossing', .34, False),
        ('contact BEFORE crossing', .30, True),
        ('contact AT crossing', .32, False),
    ]:
        p = env_module.ScenarioParams(
            seed=1, home=(start_x, 0.), v_interceptor=1., dt=.25,
            horizon_T=.25, kill_radius=.12,
            manual_threats=[dict(t=0., pos=(.1, 0.), vel=(-.5, 0.))],
        )
        env = env_module.SimEnv(p)
        event = env.step(0)  # The manual target is born before selection inside step.
        success = fixed_success if fixed else True
        assert env.intercepted == int(success), (title, env.intercepted)
        assert env.escaped == int(not success), (title, env.escaped)
        assert env.spawned == 1 and not env.active_threats()
        assert event['intercept'] == int(success)
        assert event['escape'] == int(not success)
        assert env.t == .25
        np.testing.assert_allclose(env.interceptor_pos, [start_x-.25, 0.], atol=1e-12)
        if fixed:
            assert getattr(env, 'rejected_late_intercepts', 0) == int(not success)
        print(f'PASS: {title}: intercepted={env.intercepted}, escaped={env.escaped}')
    if fixed:
        hit = env_module._boundary_fix_contact_time(
            np.array([0., 0.]), np.array([.1, 0.]), np.array([.5, 0.]),
            np.array([-.5, 0.]), 1., .15, 1.,
        )
        assert abs(hit-.5) < 1e-10
        assert not env_module._boundary_fix_valid_capture(
            np.array([0., 0.]), np.array([0., 0.]), np.array([-.01, 0.]),
            np.array([-.5, 0.]), 1., .12, .25,
        )
        print('PASS: midstep stop and already-crossed target')
    print(f'ALL ANALYTIC CHECKS PASSED ({args.version})', flush=True)

if __name__ == '__main__':
    main()
