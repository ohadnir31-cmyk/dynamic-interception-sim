from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ScenarioParams:
    seed: int = 0
    horizon_T: float = 60.0
    dt: float = 0.5

    # Stochastic generator parameters.
    lambda_arrival: float = 0.12
    x_spawn_mean: float = 30.0
    x_spawn_std: float = 10.0
    y_spawn_sigma: float = 25.0
    v_threat_mean: float = 18.0
    v_threat_std: float = 3.0

    # Larger-experiment controls.
    initial_targets: int = 0
    arrival_process: str = "bernoulli"  # bernoulli, poisson, bursty
    spatial_structure: str = "uniform"  # uniform, clustered
    n_clusters: int = 3
    cluster_std: float = 5.0
    burst_probability: float = 0.0
    burst_size_min: int = 3
    burst_size_max: int = 8

    # Metadata.
    scenario_regime: str = "unspecified"
    deadline_pressure: str = "unspecified"

    # Interceptor.
    v_interceptor: float = 20.0
    kill_radius: float = 2.0
    home: Tuple[float, float] = (0.0, 0.0)

    # Manual scenario.
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
    x = float(pos[0])
    vx = float(vel[0])
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
    Discrete-time dynamic interception environment.

    Protected boundary: x = 0.
    A target penetrates the boundary if x <= 0.
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

        self.rng_arrival = np.random.default_rng(params.seed + 101)
        self.rng_spawn = np.random.default_rng(params.seed + 202)
        self.rng_speed = np.random.default_rng(params.seed + 303)
        self.rng_angle = np.random.default_rng(params.seed + 404)
        self.rng_cluster = np.random.default_rng(params.seed + 505)
        self.rng_burst = np.random.default_rng(params.seed + 606)

        self.cluster_centers: List[Tuple[float, float]] = []
        if params.manual_threats is None and params.spatial_structure == "clustered":
            self.cluster_centers = self._sample_cluster_centers()

        self.manual_queue: List[Dict[str, Any]] = []
        if params.manual_threats is not None:
            self.manual_queue = sorted(params.manual_threats, key=lambda x: x["t"])
        else:
            self._spawn_initial_targets()

    def active_threats(self) -> List[Threat]:
        return [th for th in self.threats if not th.intercepted and not th.escaped]

    def _sample_cluster_centers(self) -> List[Tuple[float, float]]:
        centers: List[Tuple[float, float]] = []
        n_clusters = max(1, int(self.p.n_clusters))

        for _ in range(n_clusters):
            cx = max(1.0, self.rng_cluster.normal(self.p.x_spawn_mean, self.p.x_spawn_std))
            cy = self.rng_cluster.normal(0.0, max(1e-9, self.p.y_spawn_sigma))
            centers.append((float(cx), float(cy)))

        return centers

    def _sample_stochastic_position(self) -> np.ndarray:
        if self.p.spatial_structure == "clustered" and self.cluster_centers:
            cx, cy = self.cluster_centers[int(self.rng_spawn.integers(0, len(self.cluster_centers)))]
            x0 = max(1.0, self.rng_spawn.normal(cx, self.p.cluster_std))
            y0 = self.rng_spawn.normal(cy, self.p.cluster_std)
        else:
            x0 = max(1.0, self.rng_spawn.normal(self.p.x_spawn_mean, self.p.x_spawn_std))
            y0 = self.rng_spawn.normal(0.0, self.p.y_spawn_sigma)

        return np.array([x0, y0], dtype=float)

    def _sample_stochastic_velocity(self) -> np.ndarray:
        speed = max(1e-6, self.rng_speed.normal(self.p.v_threat_mean, self.p.v_threat_std))
        theta = self.rng_angle.normal(0.0, 0.25)

        vx = -speed * np.cos(theta)
        vy = speed * np.sin(theta)

        return np.array([vx, vy], dtype=float)

    def _add_stochastic_threat(self, t_birth: Optional[float] = None) -> None:
        th = Threat(
            id=self.next_threat_id,
            pos=self._sample_stochastic_position(),
            vel=self._sample_stochastic_velocity(),
            t_birth=float(self.t if t_birth is None else t_birth),
        )

        self.threats.append(th)
        self.next_threat_id += 1
        self.spawned += 1

    def _spawn_initial_targets(self) -> None:
        for _ in range(max(0, int(self.p.initial_targets))):
            self._add_stochastic_threat(t_birth=0.0)

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

    def _stochastic_arrival_count(self) -> int:
        process = (self.p.arrival_process or "bernoulli").lower()
        mean_count = max(0.0, self.p.lambda_arrival * self.p.dt)

        if process == "poisson":
            return int(self.rng_arrival.poisson(mean_count))

        if process == "bursty":
            base_count = int(self.rng_arrival.poisson(mean_count))
            burst_count = 0

            if self.rng_burst.random() < max(0.0, self.p.burst_probability) * self.p.dt:
                low = max(1, int(self.p.burst_size_min))
                high = max(low, int(self.p.burst_size_max))
                burst_count = int(self.rng_burst.integers(low, high + 1))

            return base_count + burst_count

        # Backward-compatible behavior: at most one arrival per time step.
        return int(self.rng_arrival.random() < mean_count)

    def _spawn_stochastic_if_due(self) -> int:
        arrivals = self._stochastic_arrival_count()
        for _ in range(arrivals):
            self._add_stochastic_threat()
        return arrivals

    def step(self, target_id: Optional[int]) -> Dict[str, int]:
        events = {"arrival": 0, "intercept": 0, "escape": 0}

        # 1. Arrivals.
        if self.p.manual_threats is not None:
            events["arrival"] += self._spawn_manual_threats_due()
        else:
            events["arrival"] += self._spawn_stochastic_if_due()

        # 2. Target motion.
        for th in self.active_threats():
            th.pos = th.pos + th.vel * self.p.dt

        # 3. Boundary penetration.
        for th in self.active_threats():
            if th.pos[0] <= 0.0:
                th.escaped = True
                self.escaped += 1
                events["escape"] += 1

        # 4. Interceptor motion.
        active = self.active_threats()
        target = None

        if target_id is not None:
            for th in active:
                if th.id == target_id:
                    target = th
                    break

        if target is None:
            destination = np.array(self.p.home, dtype=float)
        else:
            destination = target.pos

        self.interceptor_pos = move_toward(
            self.interceptor_pos,
            destination,
            self.p.v_interceptor,
            self.p.dt,
        )

        # 5. Interception.
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
