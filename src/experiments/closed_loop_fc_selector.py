from __future__ import annotations

"""Closed-loop deployment of the selector learned from the existing
fixed-continuation rollout dataset.

The existing dataset estimates, for every sampled state, the continuation
performance/regret of each heuristic when that heuristic is held fixed for the
remaining horizon.  A selector trained on those labels is denoted ``mu_FC``.

This module does *not* generate new adaptive rollout labels.  Instead, it makes
the minimal next methodological step:

1. train ``mu_FC`` from the already generated fixed-continuation dataset;
2. invoke ``mu_FC`` again whenever the currently pursued target is resolved;
3. let the newly selected heuristic choose the next target.

A pursued target is resolved when it is intercepted or crosses the protected
boundary.  New arrivals and events involving other targets change the next
state, but do not pre-empt the current pursuit.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.experiments.evaluate_regret_selector import (
    BASE_OBSERVABLE_FEATURES,
    OPTIONAL_OBSERVABLE_FEATURES,
    add_active_bucket,
    add_model_outcomes,
    clean_feature_values,
    compute_winner_margin,
    load_dataset,
    make_sample_weights,
    predict_regrets,
    split_by_scenario,
    summarize_by_bucket,
    summarize_model_fit,
    summarize_selector,
)
from src.experiments.rollout_labeling import extract_state_features
from src.sim.env import SimEnv, ScenarioParams
from src.sim.heuristics import make_heuristics


# The historical dataset and original code use NI.  The proposal now displays
# the same rule as NT.  Keeping NI internally avoids breaking existing CSVs.
DEFAULT_CANDIDATE_HEURISTICS: tuple[str, ...] = (
    "NI",
    "FNI",
    "FMTTB",
    "MPS",
    "FCluster",
)

DISPLAY_NAME: Dict[str, str] = {
    "NI": "NT",
    "NT": "NT",
    "FNI": "FNI",
    "FMTTB": "FMTTB",
    "MPS": "MPS",
    "FCluster": "FCluster",
}

MODEL_SCHEMA_VERSION = 1


def display_heuristic_name(name: str) -> str:
    return DISPLAY_NAME.get(str(name), str(name))


def canonical_code_name(name: str) -> str:
    """Normalize the proposal's NT label to the historical NI code label."""
    return "NI" if str(name) == "NT" else str(name)


def infer_dataset_heuristics(df: pd.DataFrame) -> List[str]:
    """Infer the available fixed-continuation outcome columns.

    The expected current dataset uses NI, FNI, FMTTB, MPS, and FCluster.
    A future dataset that uses NT instead of NI is also accepted.
    """

    found: List[str] = []
    for requested in DEFAULT_CANDIDATE_HEURISTICS:
        candidates = [requested]
        if requested == "NI":
            candidates.append("NT")

        matched = None
        for candidate in candidates:
            required = [
                f"{candidate}_future_intercepted",
                f"{candidate}_regret",
                f"{candidate}_rank",
            ]
            if all(column in df.columns for column in required):
                matched = candidate
                break

        if matched is not None:
            found.append(matched)

    if not found:
        raise ValueError(
            "No fixed-continuation heuristic outcome columns were found in the dataset."
        )
    return found


def resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    missing = [f for f in BASE_OBSERVABLE_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required observable feature columns: {missing}")
    optional = [f for f in OPTIONAL_OBSERVABLE_FEATURES if f in df.columns]
    return list(BASE_OBSERVABLE_FEATURES) + optional


def _clean_full_dataset(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    clip_abs: float,
) -> tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """Clean the full fitting dataset and return the learned imputations."""

    cleaned = df.copy()
    medians: Dict[str, float] = {}
    report_rows: List[Dict[str, Any]] = []

    for feature in feature_columns:
        raw = pd.to_numeric(cleaned[feature], errors="coerce")
        array = raw.to_numpy(dtype=float)
        bad_count = int((~np.isfinite(array)).sum())

        values = raw.replace([np.inf, -np.inf], np.nan)
        median = values.median()
        if not np.isfinite(median):
            median = 0.0
        values = values.fillna(float(median))
        if clip_abs > 0:
            values = values.clip(-clip_abs, clip_abs)

        cleaned[feature] = values.astype(float)
        medians[feature] = float(median)
        report_rows.append(
            {
                "feature": feature,
                "non_finite_or_nan_count": bad_count,
                "median_imputation_value": float(median),
                "min_after_cleaning": float(cleaned[feature].min()),
                "max_after_cleaning": float(cleaned[feature].max()),
            }
        )

    matrix = cleaned[list(feature_columns)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Full feature matrix still contains non-finite values after cleaning.")

    return cleaned, medians, pd.DataFrame(report_rows)


def _fit_regret_models(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    heuristics: Sequence[str],
    *,
    random_state: int,
    sample_weights: Optional[np.ndarray],
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: Optional[int],
) -> Dict[str, RandomForestRegressor]:
    x = df[list(feature_columns)].to_numpy(dtype=float)
    models: Dict[str, RandomForestRegressor] = {}

    for offset, heuristic in enumerate(heuristics):
        y = df[f"{heuristic}_regret"].to_numpy(dtype=float)
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            max_depth=max_depth,
            random_state=int(random_state) + offset,
            n_jobs=-1,
        )
        model.fit(x, y, sample_weight=sample_weights)
        models[str(heuristic)] = model

    return models


class HeuristicSelector(Protocol):
    name: str

    def choose_heuristic(self, env: SimEnv) -> str:
        """Choose one heuristic at the current decision epoch."""


@dataclass(frozen=True)
class FixedHeuristicSelector:
    heuristic_name: str
    name: str = field(init=False)

    def __post_init__(self) -> None:
        code_name = canonical_code_name(self.heuristic_name)
        available = make_heuristics(seed=0)
        if code_name not in available:
            raise ValueError(f"Unknown heuristic: {self.heuristic_name!r}")
        object.__setattr__(self, "heuristic_name", code_name)
        object.__setattr__(
            self,
            "name",
            f"Always {display_heuristic_name(code_name)}",
        )

    def choose_heuristic(self, env: SimEnv) -> str:
        del env
        return self.heuristic_name


@dataclass
class FixedContinuationRegretSelector:
    """The learned selector ``mu_FC`` trained on the existing dataset.

    The model predicts the fixed-continuation regret associated with each
    heuristic from the currently observable state.  At deployment it is invoked
    again after every pursued-target resolution, which turns the state mapping
    into a closed-loop adaptive switching policy.
    """

    models: Mapping[str, Any]
    feature_columns: Sequence[str]
    medians: Mapping[str, float]
    candidate_heuristics: Sequence[str]
    clip_abs: float = 1_000_000.0
    name: str = "Adaptive mu_FC selector"

    def __post_init__(self) -> None:
        self.candidate_heuristics = [str(h) for h in self.candidate_heuristics]
        missing = [h for h in self.candidate_heuristics if h not in self.models]
        if missing:
            raise ValueError(f"Missing fitted regret models: {missing}")

        # Forests are trained in parallel, but closed-loop inference evaluates
        # one state at a time.  Using all CPU cores for each single-row predict
        # call is dramatically slower because joblib startup dominates.
        for model in self.models.values():
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1

    def _feature_vector(self, env: SimEnv) -> np.ndarray:
        raw_features = extract_state_features(env)
        values: List[float] = []

        for feature in self.feature_columns:
            raw_value = raw_features.get(feature, np.nan)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = np.nan

            if not np.isfinite(value):
                value = float(self.medians.get(feature, 0.0))
            if self.clip_abs > 0:
                value = float(np.clip(value, -self.clip_abs, self.clip_abs))
            values.append(value)

        return np.asarray([values], dtype=float)

    def predicted_regrets(self, env: SimEnv) -> Dict[str, float]:
        x = self._feature_vector(env)
        return {
            heuristic: float(np.asarray(self.models[heuristic].predict(x))[0])
            for heuristic in self.candidate_heuristics
        }

    def choose_heuristic(self, env: SimEnv) -> str:
        predicted = self.predicted_regrets(env)
        return min(
            self.candidate_heuristics,
            key=lambda h: (predicted[h], self.candidate_heuristics.index(h)),
        )

    def save(self, path: Path, metadata: Optional[Mapping[str, Any]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "models": dict(self.models),
                "feature_columns": list(self.feature_columns),
                "medians": dict(self.medians),
                "candidate_heuristics": list(self.candidate_heuristics),
                "clip_abs": float(self.clip_abs),
                "name": self.name,
                "metadata": dict(metadata or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> tuple["FixedContinuationRegretSelector", Dict[str, Any]]:
        bundle = joblib.load(Path(path))
        version = int(bundle.get("schema_version", 0))
        if version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported selector bundle schema {version}; "
                f"expected {MODEL_SCHEMA_VERSION}."
            )
        selector = cls(
            models=bundle["models"],
            feature_columns=bundle["feature_columns"],
            medians=bundle["medians"],
            candidate_heuristics=bundle["candidate_heuristics"],
            clip_abs=float(bundle.get("clip_abs", 1_000_000.0)),
            name=str(bundle.get("name", "Adaptive mu_FC selector")),
        )
        return selector, dict(bundle.get("metadata", {}))


@dataclass
class TrainingArtifacts:
    selector: FixedContinuationRegretSelector
    validation_summary: pd.DataFrame
    validation_by_bucket: pd.DataFrame
    validation_model_fit: pd.DataFrame
    validation_predictions: pd.DataFrame
    validation_cleaning_report: pd.DataFrame
    full_cleaning_report: pd.DataFrame
    metadata: Dict[str, Any]


def train_mu_fc_from_existing_dataset(
    input_dir: Path,
    *,
    dataset_mode: str = "no_ties",
    validation_size: float = 0.25,
    random_state: int = 42,
    sample_weight_mode: str = "margin",
    weight_alpha: float = 0.25,
    clip_abs: float = 1_000_000.0,
    n_estimators: int = 400,
    min_samples_leaf: int = 5,
    max_depth: Optional[int] = None,
) -> TrainingArtifacts:
    """Train ``mu_FC`` using the already generated fixed-continuation data.

    A scenario-level split is used for an honest internal validation.  After
    validation, the final selector is refitted on the complete existing
    dataset.  No Always-NT continuation stage and no new counterfactual rollout
    generation are performed here.
    """

    df = add_active_bucket(load_dataset(Path(input_dir), dataset_mode))
    heuristics = infer_dataset_heuristics(df)
    features = resolve_feature_columns(df)

    train_df, validation_df = split_by_scenario(
        df,
        test_size=float(validation_size),
        random_state=int(random_state),
    )
    clean_train, clean_validation, validation_cleaning = clean_feature_values(
        train_df,
        validation_df,
        features,
        float(clip_abs),
    )

    validation_weights = make_sample_weights(
        clean_train,
        heuristics,
        sample_weight_mode,
        float(weight_alpha),
    )
    validation_models = _fit_regret_models(
        clean_train,
        features,
        heuristics,
        random_state=random_state,
        sample_weights=validation_weights,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )

    validation_pred = predict_regrets(
        validation_models,
        clean_validation,
        features,
        heuristics,
    )
    validation_outcomes = add_model_outcomes(
        clean_validation,
        validation_pred,
        heuristics,
    )
    validation_summary = summarize_selector(validation_outcomes)
    validation_by_bucket = summarize_by_bucket(validation_outcomes)
    validation_model_fit = summarize_model_fit(
        validation_models,
        clean_train,
        clean_validation,
        features,
        heuristics,
    )

    # Refit the final mu_FC on all available rows after the validation report
    # has been produced.  This is the model used in the new closed-loop test.
    clean_full, full_medians, full_cleaning = _clean_full_dataset(
        df,
        features,
        float(clip_abs),
    )
    full_weights = make_sample_weights(
        clean_full,
        heuristics,
        sample_weight_mode,
        float(weight_alpha),
    )
    final_models = _fit_regret_models(
        clean_full,
        features,
        heuristics,
        random_state=random_state,
        sample_weights=full_weights,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )

    selector = FixedContinuationRegretSelector(
        models=final_models,
        feature_columns=features,
        medians=full_medians,
        candidate_heuristics=heuristics,
        clip_abs=float(clip_abs),
    )

    metadata: Dict[str, Any] = {
        "dataset_mode": dataset_mode,
        "rows": int(len(df)),
        "scenarios": int(df["scenario"].nunique()),
        "validation_size": float(validation_size),
        "random_state": int(random_state),
        "sample_weight_mode": sample_weight_mode,
        "weight_alpha": float(weight_alpha),
        "n_estimators": int(n_estimators),
        "min_samples_leaf": int(min_samples_leaf),
        "max_depth": max_depth,
        "feature_columns": list(features),
        "candidate_heuristics": list(heuristics),
        "label_semantics": "fixed-continuation regret",
        "deployment_semantics": (
            "closed-loop reselection after selected target is intercepted or crosses"
        ),
    }

    return TrainingArtifacts(
        selector=selector,
        validation_summary=validation_summary,
        validation_by_bucket=validation_by_bucket,
        validation_model_fit=validation_model_fit,
        validation_predictions=validation_outcomes,
        validation_cleaning_report=validation_cleaning,
        full_cleaning_report=full_cleaning,
        metadata=metadata,
    )


@dataclass
class DecisionTransition:
    heuristic: str
    target_id: Optional[int]
    start_time: float
    end_time: float
    steps: int
    arrivals: int
    interceptions: int
    escapes: int
    termination_reason: str


def _selected_target_resolution_reason(
    env: SimEnv,
    target_id: Optional[int],
) -> Optional[str]:
    if target_id is None:
        return None
    for threat in env.threats:
        if threat.id != target_id:
            continue
        if threat.intercepted:
            return "selected_target_intercepted"
        if threat.escaped:
            return "selected_target_crossed"
        return None
    return "selected_target_unavailable"


def proposed_target_id(env: SimEnv, heuristic_name: str) -> Optional[int]:
    code_name = canonical_code_name(heuristic_name)
    heuristics = make_heuristics(seed=env.p.seed)
    if code_name not in heuristics:
        raise KeyError(f"Unknown heuristic: {heuristic_name!r}")
    return heuristics[code_name](
        env.active_threats(),
        env.interceptor_pos,
        env.p.v_interceptor,
    )


def advance_one_pursuit(
    env: SimEnv,
    heuristic_name: str,
) -> DecisionTransition:
    """Use one heuristic to choose one target and pursue it to resolution.

    The next decision epoch begins only when the selected target is intercepted,
    the selected target crosses the boundary, the selected heuristic returns no
    target (after one waiting step), or the finite horizon ends.
    """

    code_name = canonical_code_name(heuristic_name)
    target_id = proposed_target_id(env, code_name)
    start_time = float(env.t)
    steps = arrivals = interceptions = escapes = 0
    reason = "horizon"

    while not env.done():
        events = env.step(target_id)
        steps += 1
        arrivals += int(events.get("arrival", 0))
        interceptions += int(events.get("intercept", 0))
        escapes += int(events.get("escape", 0))

        if target_id is None:
            # Match the legacy fixed-heuristic behavior: wait one simulation
            # step, then allow the policy to reconsider in the updated state.
            reason = "no_target_selected"
            break

        resolution = _selected_target_resolution_reason(env, target_id)
        if resolution is not None:
            reason = resolution
            break

    return DecisionTransition(
        heuristic=code_name,
        target_id=target_id,
        start_time=start_time,
        end_time=float(env.t),
        steps=steps,
        arrivals=arrivals,
        interceptions=interceptions,
        escapes=escapes,
        termination_reason=reason,
    )


def run_closed_loop_selector(
    params: ScenarioParams,
    selector: HeuristicSelector,
    *,
    collect_decisions: bool = True,
) -> Dict[str, Any]:
    """Run a selector that is invoked after every pursued-target resolution."""

    env = SimEnv(params)
    decision_log: List[Dict[str, Any]] = []
    previous_heuristic: Optional[str] = None
    switch_count = 0
    decision_count = 0
    heuristic_counts: Dict[str, int] = {}

    while not env.done():
        if not env.active_threats():
            env.step(None)
            continue

        state_features = extract_state_features(env)
        predicted_regrets: Dict[str, float] = {}
        if hasattr(selector, "predicted_regrets"):
            # Compute the forest predictions once and reuse them for both the
            # decision and the optional trace.
            predicted_regrets = dict(getattr(selector, "predicted_regrets")(env))
            ordered = list(getattr(selector, "candidate_heuristics", predicted_regrets))
            heuristic = min(
                ordered,
                key=lambda h: (predicted_regrets[h], ordered.index(h)),
            )
        else:
            heuristic = canonical_code_name(selector.choose_heuristic(env))

        heuristic = canonical_code_name(heuristic)
        decision_count += 1
        display_name = display_heuristic_name(heuristic)
        heuristic_counts[display_name] = heuristic_counts.get(display_name, 0) + 1

        if previous_heuristic is not None and heuristic != previous_heuristic:
            switch_count += 1

        before_intercepted = int(env.intercepted)
        before_escaped = int(env.escaped)
        transition = advance_one_pursuit(env, heuristic)

        if collect_decisions:
            row: Dict[str, Any] = {
                "decision_index": len(decision_log),
                "selector": selector.name,
                "time": transition.start_time,
                "heuristic": display_name,
                "heuristic_code": heuristic,
                "selected_target_id": transition.target_id,
                "decision_end_time": transition.end_time,
                "decision_steps": transition.steps,
                "termination_reason": transition.termination_reason,
                "arrivals_during_pursuit": transition.arrivals,
                "interceptions_during_pursuit": int(env.intercepted) - before_intercepted,
                "escapes_during_pursuit": int(env.escaped) - before_escaped,
                **state_features,
            }
            for candidate, regret in predicted_regrets.items():
                row[f"{display_heuristic_name(candidate)}_predicted_regret"] = float(regret)
            decision_log.append(row)

        previous_heuristic = heuristic

    return {
        "policy": selector.name,
        "spawned": int(env.spawned),
        "intercepted": int(env.intercepted),
        "escaped": int(env.escaped),
        "interception_rate": float(env.intercepted / max(1, env.spawned)),
        "num_decisions": int(decision_count),
        "num_heuristic_switches": int(switch_count),
        "heuristic_counts": heuristic_counts,
        "decision_log": decision_log,
    }
