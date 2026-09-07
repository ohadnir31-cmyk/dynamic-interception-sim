"""Replay saved ScenarioParams using real repository functions, in a fresh process.

No locally transcribed simulator is used. Requires the proposal checkout API.
Every ScenarioParams field must be present in the saved CSV; missing values are
not silently filled with today's defaults. Model loading is for YOUR trusted
joblib files only. Existing output directories are never overwritten.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import fields
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import get_type_hints


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_saved_scenarios(path, env_module, scenario_class, limit=0):
    import pandas as pd
    frame = pd.read_csv(path, float_precision="round_trip")
    if frame.empty:
        raise ValueError('The scenario CSV is empty.')
    field_names = [f.name for f in fields(env_module.ScenarioParams)]
    missing = sorted(set(['scenario', *field_names]) - set(frame.columns))
    if missing:
        raise ValueError(
            f'Missing saved scenario fields: {missing}. '
            'Retrieve the original configuration; no defaults were substituted.'
        )
    if frame['scenario'].isna().any() or frame['scenario'].duplicated().any():
        raise ValueError('Scenario IDs must be non-null and unique.')
    if limit:
        if limit < 1 or limit > len(frame):
            raise ValueError(f'--limit must be between 1 and {len(frame)}, or 0 for all.')
        frame = frame.iloc[:limit].copy()
    hints = get_type_hints(env_module.ScenarioParams)
    scenarios = {}
    for row in frame.to_dict('records'):
        kwargs = {}
        for name in field_names:
            value = row[name]
            if name == 'manual_threats':
                if pd.isna(value) or str(value).strip() in {'', 'None', 'null'}:
                    value = None
                elif isinstance(value, str):
                    value = ast.literal_eval(value)
                if value is not None and not isinstance(value, list):
                    raise ValueError('manual_threats must be a list or null.')
            elif name == 'home':
                value = ast.literal_eval(value) if isinstance(value, str) else value
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise ValueError('Invalid saved home coordinates.')
                value = tuple(float(x) for x in value)
            else:
                if pd.isna(value):
                    raise ValueError(f'Missing value for {name} in {row["scenario"]}.')
                if hints.get(name) is int:
                    if int(value) != float(value):
                        raise ValueError(f'Noninteger saved {name}: {value}')
                    value = int(value)
                elif hints.get(name) is float:
                    value = float(value)
                elif hints.get(name) is str:
                    value = str(value)
            kwargs[name] = value
        scenarios[str(row['scenario'])] = scenario_class(params=env_module.ScenarioParams(**kwargs))
    return frame, scenarios


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--params-csv', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--mode', choices=['fixed', 'labels', 'frozen'], default='fixed')
    p.add_argument('--model-in', type=Path)
    p.add_argument('--sampling-mode', choices=['legacy_active_steps', 'decision_epochs_uniform'])
    p.add_argument('--label-scenarios', type=int)
    p.add_argument('--max-states-per-run', type=int)
    p.add_argument('--fallback', choices=['none', 'NI'])
    p.add_argument('--keep-duplicate-initial-states', action='store_true')
    args = p.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import pandas as pd
    from tqdm.auto import tqdm
    env_module = importlib.import_module('src.sim.env')
    if Path(env_module.__file__).resolve() != repo/'src/sim/env.py':
        raise RuntimeError(f'Wrong simulator imported: {env_module.__file__}')
    factory = importlib.import_module('src.experiments.run_large_scale_rollout')
    frame, scenarios = load_saved_scenarios(
        args.params_csv, env_module, factory.ScenarioObj, args.limit,
    )
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'mode': args.mode, 'status': 'started', 'repo': str(repo),
        'env_sha256': sha256(repo/'src/sim/env.py'),
        'source_params_csv': str(args.params_csv.resolve()),
        'source_params_sha256': sha256(args.params_csv),
        'n_scenarios': len(scenarios),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'scope': 'Actual repository execution; no audit_core simulator.',
    }
    (out/'replay_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    frame.to_csv(out/'large_scale_scenario_params.csv', index=False)
    print(f'Imported simulator: {env_module.__file__}', flush=True)
    print(f'Replaying {len(scenarios)} saved scenario configurations; mode={args.mode}', flush=True)
    heuristics = factory.CANDIDATE_HEURISTICS
    if args.mode == 'fixed':
        result = factory.run_full_heuristic_rollouts(
            scenarios, heuristics, out/'large_scale_full_heuristic_rollouts.csv',
        )
        factory.summarize_heuristics(result, out/'large_scale_heuristic_summary.csv')
        factory.summarize_scenario_winners(result, out/'large_scale_scenario_winners.csv')
    elif args.mode == 'labels':
        required = (args.sampling_mode, args.label_scenarios, args.max_states_per_run, args.fallback)
        if any(x is None for x in required):
            raise ValueError('Label replay requires explicit sampling-mode, label-scenarios, max-states-per-run and fallback.')
        if not 1 <= args.label_scenarios <= len(scenarios) or args.max_states_per_run < 1:
            raise ValueError('Invalid label scenario/state count.')
        label_module = importlib.import_module('src.experiments.rollout_labeling')
        selected = dict(list(scenarios.items())[:args.label_scenarios])
        df = label_module.generate_dataset_for_scenarios(
            scenarios=selected,
            behavior_heuristics=factory.DEFAULT_BEHAVIOR_HEURISTICS,
            candidate_heuristics=heuristics,
            rollout_preempt=False,
            max_states_per_run=args.max_states_per_run,
            state_sampling_mode=args.sampling_mode,
            behavior_no_target_fallback=None if args.fallback == 'none' else args.fallback,
            deduplicate_initial_states=not args.keep_duplicate_initial_states,
        )
        df.to_csv(out/'large_scale_rollout_states.csv', index=False)
        for keep, suffix in [(True, 'with_ties'), (False, 'no_ties')]:
            factory.filter_informative_states(df, heuristics, keep_ties=keep).to_csv(
                out/f'large_scale_rollout_states_informative_{suffix}.csv', index=False,
            )
        factory.summarize_by_active_targets(df, out/'large_scale_active_targets_analysis.csv')
    else:
        if args.model_in is None or not args.model_in.is_file():
            raise ValueError('--model-in must name your saved, trusted selector file.')
        selector_module = importlib.import_module('src.experiments.closed_loop_fc_selector')
        runner = importlib.import_module('src.sim.runner')
        selector, metadata = selector_module.FixedContinuationRegretSelector.load(args.model_in)
        manifest['model_sha256'] = sha256(args.model_in)
        manifest['model_frozen'] = True
        manifest['baseline_heuristic'] = selector.baseline_heuristic
        manifest['threshold'] = selector.regret_threshold
        manifest['threshold_mode'] = selector.threshold_mode
        rows = []
        for name, obj in tqdm(scenarios.items(), desc='Frozen selector and fixed baselines'):
            for h in heuristics:
                result = runner.run_episode(obj.params, h, preempt=False)
                rows.append(dict(scenario=name, policy=f'Always {h}',
                                 spawned=result['spawned'], intercepted=result['intercepted'],
                                 escaped=result['escaped']))
            for policy, fn in [('Frozen closed-loop', selector_module.run_closed_loop_selector),
                               ('Frozen one-shot', selector_module.run_one_shot_selector)]:
                result = fn(obj.params, selector, collect_decisions=False)
                rows.append(dict(scenario=name, policy=policy,
                                 spawned=result['spawned'], intercepted=result['intercepted'],
                                 escaped=result['escaped']))
            # Partial scores survive interruption; this is NOT an automatic resume facility.
            pd.DataFrame(rows).to_csv(out/'frozen_policy_results.csv', index=False)
        pd.DataFrame(rows).groupby('policy', sort=False).agg(
            mean_intercepted=('intercepted', 'mean'), mean_escaped=('escaped', 'mean'),
            n_scenarios=('scenario', 'nunique'),
        ).reset_index().to_csv(out/'frozen_policy_summary.csv', index=False)
    manifest['status'] = 'completed'
    (out/'replay_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'COMPLETED: {out}', flush=True)

if __name__ == '__main__':
    main()
