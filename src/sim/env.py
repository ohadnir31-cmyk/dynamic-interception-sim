from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np

@dataclass
class ScenarioParams:
    seed: int = 0
    horizon_T: float = 120.0
    dt: float = 0.2

    # Arrival (world generator truth)
    lambda_arrival: float = 0.12  # threats per second

    # Spawn location distribution (boundary at x=0, safe side x>0)
    x_spawn_mean: float = 60.0
    x_spawn_std: float = 10.0
    y_spawn_sigma: float = 40.0

    # Threat speed distribution (toward boundary)
    v_threat_mean: float = 12.0
    v_threat_std: float = 2.0

    # Interceptor
    v_interceptor: float = 25.0
    kill_radius: float = 2.0

    # Home / wait point
    home: Tuple[float, float] = (0.0, 0.0)

@dataclass
class Threat:
    id: int
    pos: np.ndarray  # (2,)
    vel: np.ndarray  # (2,)
    t_birth: float
    intercepted: bool = False
    escaped: bool = False

def time_to_boundary_x0(pos: np.ndarray, vel: np.ndarray) -> float:
    """TTB for boundary x=0, assuming constant velocity."""
    x, vx = float(pos[0]), float(vel[0])
    if vx >= 0:
        return np.inf
    t = (0.0 - x) / vx  # vx negative => t positive
    return t if t >= 0 else np.inf

def time_to_intercept(interceptor_pos: np.ndarray, target_pos: np.ndarray, vI: float) -> float:
    return float(np.linalg.norm(target_pos - interceptor_pos) / max(1e-9, vI))

def slack(interceptor_pos: np.ndarray, th: Threat, vI: float) -> float:
    return time_to_boundary_x0(th.pos, th.vel) - time_to_intercept(interceptor_pos, th.pos, vI)

def move_toward(p: np.ndarray, q: np.ndarray, speed: float, dt: float) -> np.ndarray:
    d = q - p
    dist = float(np.linalg.norm(d))
    if dist < 1e-12:
        return p.copy()
    step = speed * dt
    if step >= dist:
        return q.copy()
    return p + d * (step / dist)

class SimEnv:
    """
    Minimal dynamic interception environment.
    - boundary at x=0 (escape if x<=0)
    - threats spawn online
    - interceptor moves toward chosen target (or home if none)
    """
    def __init__(self, params: ScenarioParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)

        self.t = 0.0
        self.threat_id = 0
        self.threats: List[Threat] = []
        self.arrival_times: List[float] = []

        self.interceptor_pos = np.array(params.home, dtype=float)

        self.intercepted = 0
        self.escaped = 0
        self.spawned = 0

    def active_threats(self) -> List[Threat]:
        return [th for th in self.threats if (not th.intercepted) and (not th.escaped)]

    def step(self, target_id: Optional[int]) -> Dict[str, int]:
        """
        Advance by dt:
        - possibly spawn new threat
        - move threats
        - move interceptor (toward target or home)
        - resolve intercept/escape events
        Returns event counters for this step.
        """
        events = {"arrival": 0, "intercept": 0, "escape": 0}

        # spawn (discrete approx to Poisson)
        if self.rng.random() < self.p.lambda_arrival * self.p.dt:
            x0 = max(1.0, self.rng.normal(self.p.x_spawn_mean, self.p.x_spawn_std))
            y0 = self.rng.normal(0.0, self.p.y_spawn_sigma)

            speed = max(1.0, self.rng.normal(self.p.v_threat_mean, self.p.v_threat_std))
            # mostly toward -x with small angular noise
            theta = self.rng.normal(0.0, 0.25)
            vx = -speed * np.cos(theta)
            vy = speed * np.sin(theta)

            th = Threat(
                id=self.threat_id,
                pos=np.array([x0, y0], dtype=float),
                vel=np.array([vx, vy], dtype=float),
                t_birth=self.t
            )
            self.threats.append(th)
            self.arrival_times.append(self.t)
            self.threat_id += 1
            self.spawned += 1
            events["arrival"] = 1

        # move threats and check escape
        for th in self.active_threats():
            th.pos = th.pos + th.vel * self.p.dt
            if th.pos[0] <= 0.0:
                th.escaped = True
                self.escaped += 1
                events["escape"] += 1

        # move interceptor
        active = self.active_threats()
        home = np.array(self.p.home, dtype=float)

        target = None
        if target_id is not None:
            for th in active:
                if th.id == target_id:
                    target = th
                    break

        if target is None:
            # go / stay at home
            self.interceptor_pos = move_toward(self.interceptor_pos, home, self.p.v_interceptor, self.p.dt)
        else:
            self.interceptor_pos = move_toward(self.interceptor_pos, target.pos, self.p.v_interceptor, self.p.dt)

        # intercept check (after movement)
        if target is not None and (not target.escaped) and (not target.intercepted):
            if float(np.linalg.norm(target.pos - self.interceptor_pos)) <= self.p.kill_radius:
                target.intercepted = True
                self.intercepted += 1
                events["intercept"] += 1

        self.t += self.p.dt
        return events

    def done(self) -> bool:
        return self.t >= self.p.horizon_T
