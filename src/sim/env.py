from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


EPS = 1e-9


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

    # Lower bounds for stochastic target generation.
    # These avoid pathological targets that almost do not move toward x=0,
    # which create operationally meaningless TTB values in a finite-horizon
    # simulation. Manual scenarios are not affected.
    min_threat_speed: float = 0.05
    min_boundary_speed: float = 0.05

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
    """
    Time until the target reaches the protected boundary x = 0.

    Returns inf if the target is not moving toward the boundary.
    """
    x = float(pos[0])
    vx = float(vel[0])

    if vx >= 0:
        return np.inf

    t = (0.0 - x) / vx
    return float(t) if t >= 0 else np.inf


def time_to_intercept(
    interceptor_pos: np.ndarray,
    target_pos: np.ndarray,
    vI: float,
    target_vel: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate time-to-intercept.

    If target_vel is provided, this solves the constant-velocity lead-intercept
    equation:

        ||target_pos + target_vel * t - interceptor_pos|| = vI * t

    and returns the smallest non-negative solution. This is the time required
    for an interceptor flying at speed vI to meet the moving target by aiming at
    the predicted future intercept point.

    If target_vel is omitted, the function falls back to the older static
    distance / speed calculation. This fallback is kept only for backward
    compatibility with older analysis notebooks.
    """
    pI = np.array(interceptor_pos, dtype=float)
    pT = np.array(target_pos, dtype=float)
    speed = max(float(vI), EPS)

    r = pT - pI

    if target_vel is None:
        return float(np.linalg.norm(r) / speed)

    vT = np.array(target_vel, dtype=float)

    c = float(np.dot(r, r))
    if c <= EPS:
        return 0.0

    a = float(np.dot(vT, vT) - speed * speed)
    b = float(2.0 * np.dot(r, vT))

    # Linear case: a ~= 0.
    if abs(a) <= EPS:
        if abs(b) <= EPS:
            return np.inf
        t = -c / b
        return float(t) if t >= 0 else np.inf

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return np.inf

    sqrt_disc = float(np.sqrt(max(0.0, disc)))
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    nonnegative_roots = [float(t) for t in roots if t >= -EPS]

    if not nonnegative_roots:
        return np.inf

    return max(0.0, min(nonnegative_roots))


def predicted_intercept_point(
    interceptor_pos: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    vI: float,
) -> np.ndarray:
    """
    Predicted lead-intercept point for a moving target.

    If no finite lead-intercept solution exists, fall back to the target's
    current position. This keeps the simulator well-defined even for cases in
    which the target is too fast or geometrically unreachable.
    """
    tti = time_to_intercept(
        interceptor_pos=interceptor_pos,
        target_pos=target_pos,
        vI=vI,
        target_vel=target_vel,
    )

    if not np.isfinite(tti):
        return np.array(target_pos, dtype=float).copy()

    return np.array(target_pos, dtype=float) + np.array(target_vel, dtype=float) * tti


def slack(interceptor_pos: np.ndarray, th: Threat, vI: float) -> float:
    """
    Feasibility margin for a single moving target.

    slack = TTB - TTI

    where TTB is time-to-boundary and TTI is the moving-target lead-intercept
    time. A non-negative slack means that, under the lead-intercept model, the
    interceptor can reach the target before the target reaches x = 0.
    """
    return time_to_boundary_x0(th.pos, th.vel) - time_to_intercept(
        interceptor_pos=interceptor_pos,
        target_pos=th.pos,
        target_vel=th.vel,
        vI=vI,
    )


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

    Interceptor guidance model:
        When a target is assigned, the interceptor steers toward the predicted
        moving-target lead-intercept point rather than toward the target's
        current position. The guidance point is recomputed at every simulation
        step for the currently assigned target.
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
        """Sample a stochastic target velocity directed toward x=0.

        Earlier versions sampled the speed from a normal distribution and then
        applied max(1e-6, speed). Rare negative draws therefore became nearly
        stationary targets with boundary_speed close to zero. Such targets are
        mathematically valid but create huge time-to-boundary values that are
        not meaningful for the finite-horizon experiments.

        The current sampler enforces two lower bounds:
        - total target speed is at least min_threat_speed;
        - motion toward the protected boundary is at least min_boundary_speed.

        This preserves the intended interpretation that stochastic targets move
        toward the protected boundary, while avoiding pathological TTB outliers.
        """

        min_speed = max(float(self.p.min_threat_speed), EPS)
        min_boundary_speed = max(float(self.p.min_boundary_speed), EPS)

        # Try rejection sampling first so that typical velocities preserve the
        # sampled speed and heading distribution.
        for _ in range(50):
            speed = float(self.rng_speed.normal(self.p.v_threat_mean, self.p.v_threat_std))
            speed = max(speed, min_speed, min_boundary_speed)
            theta = float(self.rng_angle.normal(0.0, 0.25))

            vx = -speed * np.cos(theta)
            vy = speed * np.sin(theta)

            if -vx >= min_boundary_speed:
                return np.array([vx, vy], dtype=float)

        # Extremely unlikely fallback for very large sampled headings.
        # Keep the lateral component but enforce minimum boundary progress.
        speed = max(float(self.rng_speed.normal(self.p.v_threat_mean, self.p.v_threat_std)), min_speed, min_boundary_speed)
        theta = float(self.rng_angle.normal(0.0, 0.25))
        vx = -max(speed * np.cos(theta), min_boundary_speed)
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

    def _select_target_object(self, target_id: Optional[int]) -> Optional[Threat]:
        if target_id is None:
            return None

        for th in self.active_threats():
            if th.id == target_id:
                return th

        return None

    def _interceptor_destination(self, target: Optional[Threat]) -> np.ndarray:
        if target is None:
            return np.array(self.p.home, dtype=float)

        return predicted_intercept_point(
            interceptor_pos=self.interceptor_pos,
            target_pos=target.pos,
            target_vel=target.vel,
            vI=self.p.v_interceptor,
        )

    def step(self, target_id: Optional[int]) -> Dict[str, int]:
        events = {"arrival": 0, "intercept": 0, "escape": 0}

        # 1. Arrivals at the current time.
        if self.p.manual_threats is not None:
            events["arrival"] += self._spawn_manual_threats_due()
        else:
            events["arrival"] += self._spawn_stochastic_if_due()

        # 2. Lead-pursuit interceptor guidance based on the current state.
        target = self._select_target_object(target_id)
        destination = self._interceptor_destination(target)

        new_interceptor_pos = move_toward(
            self.interceptor_pos,
            destination,
            self.p.v_interceptor,
            self.p.dt,
        )

        # 3. Advance active threats and interceptor over this time step.
        #    This is a discrete-time approximation of simultaneous motion.
        for th in self.active_threats():
            th.pos = th.pos + th.vel * self.p.dt

        self.interceptor_pos = new_interceptor_pos

        # 4. Interceptions after the simultaneous movement update.
        victims = []
        for th in self.active_threats():
            if float(np.linalg.norm(th.pos - self.interceptor_pos)) <= self.p.kill_radius:
                victims.append(th)

        for th in victims:
            th.intercepted = True
            self.intercepted += 1
            events["intercept"] += 1

        # 5. Boundary penetration for remaining active threats.
        for th in self.active_threats():
            if th.pos[0] <= 0.0:
                th.escaped = True
                self.escaped += 1
                events["escape"] += 1

        self.t += self.p.dt
        return events

    def done(self) -> bool:
        return self.t >= self.p.horizon_T
