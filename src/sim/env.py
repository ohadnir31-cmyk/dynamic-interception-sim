from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
import numpy as np


@dataclass
class ScenarioParams:
    seed: int = 0
    horizon_T: float = 60.0
    dt: float = 0.5

    # stochastic generator (used only if manual_threats is None)
    lambda_arrival: float = 0.12
    x_spawn_mean: float = 30.0
    x_spawn_std: float = 10.0
    y_spawn_sigma: float = 25.0
    v_threat_mean: float = 18.0
    v_threat_std: float = 3.0

    # interceptor
    v_interceptor: float = 20.0
    kill_radius: float = 2.0
    home: Tuple[float, float] = (0.0, 0.0)

    # optional manual scenario
    # each threat dict:
    # {"t": 0.0, "pos": [x, y], "vel": [vx, vy]}
    manual_threats: Optional[List[Dict[str, Any]]] = None


@dataclass
class Threat:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    t_birth: float
    intercepted: bool = False
    escaped: bool = False


def time_to_boundary_x0(pos: np.ndarray, vel: np.ndarray) -> float:
    x, vx = float(pos[0]), float(vel[0])
    if vx >= 0:
        return np.inf
    t = (0.0 - x) / vx
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
    DT-based dynamic interception environment.
    Boundary is x=0. Threat escapes if x<=0.
    If manual_threats is provided, stochastic spawning is disabled.
    """

    def __init__(self, params: ScenarioParams):
        self.p = params
        self.t = 0.0

        self.interceptor_pos = np.array(params.home, dtype=float)

        self.threats: List[Threat] = []
        self.next_threat_id = 0

        self.spawned = 0
        self.intercepted = 0
        self.escaped = 0

        # separate RNG streams for reproducibility
        self.rng_arrival = np.random.default_rng(params.seed + 101)
        self.rng_spawn = np.random.default_rng(params.seed + 202)
        self.rng_speed = np.random.default_rng(params.seed + 303)
        self.rng_angle = np.random.default_rng(params.seed + 404)

        # manual threat schedule
        self.manual_queue: List[Dict[str, Any]] = []
        if params.manual_threats is not None:
            self.manual_queue = sorted(params.manual_threats, key=lambda x: x["t"])

    def active_threats(self) -> List[Threat]:
        return [th for th in self.threats if (not th.intercepted) and (not th.escaped)]

    def _spawn_manual_threats_due(self) -> int:
        arrivals = 0
        eps = 1e-9
        while self.manual_queue and self.manual_queue[0]["t"] <= self.t + eps:
            item = self.manual_queue.pop(0)
            th = Threat(
                id=self.next_threat_id,
                pos=np.array(item["pos"], dtype=float),
                vel=np.array(item["vel"], dtype=float),
                t_birth=float(item["t"]),
            )
            self.threats.append(th)
            self.next_threat_id += 1
            self.spawned += 1
            arrivals += 1
        return arrivals

    def _spawn_stochastic_if_due(self) -> int:
        arrivals = 0
        if self.rng_arrival.random() < self.p.lambda_arrival * self.p.dt:
            x0 = max(1.0, self.rng_spawn.normal(self.p.x_spawn_mean, self.p.x_spawn_std))
            y0 = self.rng_spawn.normal(0.0, self.p.y_spawn_sigma)

            speed = max(1.0, self.rng_speed.normal(self.p.v_threat_mean, self.p.v_threat_std))
            theta = self.rng_angle.normal(0.0, 0.25)

            vx = -speed * np.cos(theta)
            vy = speed * np.sin(theta)

            th = Threat(
                id=self.next_threat_id,
                pos=np.array([x0, y0], dtype=float),
                vel=np.array([vx, vy], dtype=float),
                t_birth=self.t,
            )
            self.threats.append(th)
            self.next_threat_id += 1
            self.spawned += 1
            arrivals += 1
        return arrivals

    def step(self, target_id: Optional[int]) -> Dict[str, int]:
        events = {"arrival": 0, "intercept": 0, "escape": 0}

        # arrivals
        if self.p.manual_threats is not None:
            events["arrival"] += self._spawn_manual_threats_due()
        else:
            events["arrival"] += self._spawn_stochastic_if_due()

        # move threats
        for th in self.active_threats():
            th.pos = th.pos + th.vel * self.p.dt

        # threats may escape in same step they spawned
        for th in self.active_threats():
            if th.pos[0] <= 0.0:
                th.escaped = True
                self.escaped += 1
                events["escape"] += 1

        # move interceptor
        active = self.active_threats()
        target = None
        if target_id is not None:
            for th in active:
                if th.id == target_id:
                    target = th
                    break

        if target is None:
            home = np.array(self.p.home, dtype=float)
            self.interceptor_pos = move_toward(self.interceptor_pos, home, self.p.v_interceptor, self.p.dt)
        else:
            self.interceptor_pos = move_toward(self.interceptor_pos, target.pos, self.p.v_interceptor, self.p.dt)

        # multi-kill: all active threats within kill_radius are intercepted
        victims = []
        for th in self.active_threats():
            if float(np.linalg.norm(th.pos - self.interceptor_pos)) <= self.p.kill_radius:
                victims.append(th)

        for th in victims:
            th.intercepted = True
            self.intercepted += 1
            events["intercept"] += 1

        self.t += self.p.dt
        return events

    def done(self) -> bool:
        return self.t >= self.p.horizon_T
