#!/usr/bin/env python3
"""Compare original/fixed scoring on full fixed-heuristic scenarios.

Run from the repository root. Only src/sim/env.py may differ semantically
from BASE in src/. The local repository and old outputs are never modified.
This runner neither trains models nor generates state labels.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

BASE = '2a4f1df6449443fd6f0ade8d97b11fb65876bf7c'
TOOLS = Path(__file__).resolve().parent
PARAMS_FILE = 'large_scale_scenario_params.csv'
RESULTS_FILE = 'large_scale_full_heuristic_rollouts.csv'


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(repo), *args])


def normalized(data: bytes) -> bytes:
    return data.replace(b'\r\n', b'\n')


def code_equal(left: str, right: str) -> bool:
    """Ignore comments/formatting, but not executable code or docstrings."""
    return ast.dump(ast.parse(left)) == ast.dump(ast.parse(right))


def check_source(repo: Path) -> tuple[str, str]:
    if Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != repo:
        raise ValueError('--repo must be the actual repository root.')
    git(repo, 'cat-file', '-e', BASE + '^{commit}')
    base_env = git(repo, 'show', BASE + ':src/sim/env.py').decode('utf-8')
    current = (repo/'src/sim/env.py').read_text(encoding='utf-8')
    spec = importlib.util.spec_from_file_location('_local_boundary_patch', TOOLS/'apply_boundary_fix.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.MARKER not in current:
        raise ValueError('src/sim/env.py is not patched. Install the fix first.')
    if not code_equal(current, module.patched_text(base_env)):
        raise ValueError('env.py includes changes beyond this narrow fix. Stop and review the diff.')
    names = git(repo, 'ls-tree', '-r', '--name-only', BASE, '--', 'src', 'requirements.txt').decode().splitlines()
    expected_py = {name for name in names if name.endswith('.py')}
    actual_py = {str(p.relative_to(repo)).replace(os.sep, '/') for p in (repo/'src').rglob('*.py')}
    if actual_py != expected_py:
        raise ValueError('The src Python file list differs from the proposal snapshot.')
    for name in names:
        if name == 'src/sim/env.py':
            continue
        local = repo/name
        if not local.is_file() or local.is_symlink():
            raise ValueError(f'Missing or unsupported local source file: {name}')
        original = git(repo, 'show', BASE + ':' + name)
        if normalized(local.read_bytes()) != normalized(original):
            raise ValueError(f'Additional source change in {name}. Do not mix it into this scoring comparison.')
    return base_env, current


def extract_source(repo: Path, destination: Path) -> None:
    """Extract only tracked source/tests/dependencies, rejecting unsafe entries."""
    data = git(repo, 'archive', '--format=tar', BASE, 'src', 'tests', 'requirements.txt')
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe archive path.')
            target = destination/name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
            else:
                raise ValueError('Links/devices are not accepted in the source archive.')


def run(command: list, cwd: Path, log: Path) -> None:
    command = list(map(str, command))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(cwd)
    env['PYTHONUNBUFFERED'] = '1'
    print('\nCOMMAND:', ' '.join(command), '\nCWD:', cwd, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        stream.write('CWD: ' + str(cwd) + '\nCOMMAND: ' + json.dumps(command) + '\n')
        stream.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if return_code:
        raise RuntimeError(f'Command failed (exit {return_code}). Inspect {log}')


def load_scores(path: Path):
    import pandas as pd
    frame = pd.read_csv(path, float_precision='round_trip')
    frame = frame.rename(columns={'penetrated': 'escaped'})
    required = ['scenario', 'heuristic', 'spawned', 'intercepted', 'escaped']
    if not set(required) <= set(frame):
        raise ValueError(f'Missing result columns in {path}: {set(required)-set(frame)}')
    frame = frame[required].copy()
    if frame.empty or frame.isna().any().any() or frame.duplicated(['scenario', 'heuristic']).any():
        raise ValueError('Empty/duplicate/missing score rows.')
    return frame


def verify_historical(replayed: Path, historical: Path) -> None:
    import pandas as pd
    current = load_scores(replayed).set_index(['scenario', 'heuristic']).sort_index()
    previous = load_scores(historical).set_index(['scenario', 'heuristic']).sort_index()
    if len(current.index.difference(previous.index)):
        raise ValueError('The historical scores do not include every replayed scenario/heuristic.')
    pd.testing.assert_frame_equal(current, previous.loc[current.index], check_dtype=False, check_exact=True)
    print('ORIGINAL REPLAY MATCHES THE SAVED SCORES EXACTLY.', flush=True)


def compare_results(original_dir: Path, fixed_dir: Path, output: Path):
    import numpy as np
    import pandas as pd
    old_params = pd.read_csv(original_dir/PARAMS_FILE, float_precision='round_trip')
    new_params = pd.read_csv(fixed_dir/PARAMS_FILE, float_precision='round_trip')
    pd.testing.assert_frame_equal(old_params, new_params, check_dtype=False, check_exact=True)
    old, new = load_scores(original_dir/RESULTS_FILE), load_scores(fixed_dir/RESULTS_FILE)
    pair = old.merge(new, on=['scenario', 'heuristic'], how='outer',
                     suffixes=('_original', '_fixed'), indicator=True, validate='one_to_one')
    if not pair['_merge'].eq('both').all():
        raise ValueError('The two runs do not cover exactly the same scenario/heuristic pairs.')
    pair = pair.drop(columns='_merge')
    removed = pair['intercepted_original'] - pair['intercepted_fixed']
    if not np.array_equal(pair['spawned_original'], pair['spawned_fixed']):
        raise ValueError('Spawned target counts changed.')
    if (removed < 0).any():
        raise ValueError('Fixed-heuristic captures increased; this is not the expected narrow scoring change.')
    if not np.array_equal(removed, pair['escaped_fixed']-pair['escaped_original']):
        raise ValueError('The capture decrease does not match the crossing increase.')
    pair['removed'] = removed
    pair['fixed_minus_original'] = -removed
    pair['affected'] = removed.ne(0)
    summary = pair.groupby('heuristic', sort=False).agg(
        n_scenarios=('scenario', 'nunique'),
        original_mean=('intercepted_original', 'mean'),
        fixed_mean=('intercepted_fixed', 'mean'),
        removed_mean=('removed', 'mean'), removed_total=('removed', 'sum'),
        affected_scenarios=('affected', 'sum'),
    ).reset_index().sort_values('original_mean', ascending=False)
    pair.to_csv(output/'paired_results.csv', index=False)
    summary.to_csv(output/'boundary_fix_comparison.csv', index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path, required=True, help='A NEW directory; no overwrite.')
    parser.add_argument('--n-scenarios', type=int, default=20, help='New pilot scenarios only.')
    parser.add_argument('--seed', type=int, default=20260906, help='Seed for a NEW pilot, not a historical claim.')
    parser.add_argument('--params-csv', type=Path, help='Saved historical scenario parameters instead of a new pilot.')
    parser.add_argument('--limit', type=int, default=0, help='Saved replay: first N; 0 means all.')
    parser.add_argument('--original-results', type=Path, help='Optional historical scores: require exact original replay match.')
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output_dir.resolve()
    if args.n_scenarios < 1 or args.limit < 0:
        parser.error('Counts must be valid nonnegative integers; n-scenarios must be positive.')
    if not args.params_csv and (args.original_results or args.limit):
        parser.error('--original-results/--limit require --params-csv.')
    for path in [args.params_csv, args.original_results]:
        if path and not path.is_file():
            parser.error(f'File not found: {path}')
    for name in ['apply_boundary_fix.py', 'test_real_checkout.py', 'replay_saved_scenarios.py']:
        if not (TOOLS/name).is_file():
            parser.error(f'Missing helper: {TOOLS/name}')
    if out.exists():
        parser.error(f'Output exists. Choose a NEW output directory: {out}')
    base_text, fixed_text = check_source(repo)
    out.mkdir(parents=True)
    (out/'logs').mkdir()
    (out/'code').mkdir()
    (out/'code/env_original.py').write_text(base_text, encoding='utf-8')
    (out/'code/env_fixed.py').write_text(fixed_text, encoding='utf-8')
    manifest = {
        'status': 'started', 'utc_started': datetime.now(timezone.utc).isoformat(),
        'baseline_commit': BASE, 'checkout_commit': git(repo, 'rev-parse', 'HEAD').decode().strip(),
        'python': sys.version, 'working_tree': git(repo, 'status', '--porcelain').decode(),
        'fixed_env_sha256': hashlib.sha256(fixed_text.encode()).hexdigest(),
        'scope': 'Full scenarios; fixed heuristics only; narrow scoring fix. No labels or training.',
        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    manifest_path = out/'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        (out/'environment_freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
        with tempfile.TemporaryDirectory(prefix='boundary_pair_') as work:
            original = Path(work)/'original'
            fixed = Path(work)/'fixed'
            extract_source(repo, original)
            shutil.copytree(original, fixed)
            (fixed/'src/sim/env.py').write_text(fixed_text, encoding='utf-8')
            for version, root in [('original', original), ('fixed', fixed)]:
                run([sys.executable, TOOLS/'test_real_checkout.py', '--repo', root, '--version', version],
                    root, out/'logs'/f'analytic_{version}.log')
                run([sys.executable, '-m', 'pytest', '-q', 'tests'],
                    root, out/'logs'/f'pytest_{version}.log')
            original_output, fixed_output = out/'original', out/'fixed'
            if args.params_csv:
                run([sys.executable, TOOLS/'replay_saved_scenarios.py', '--repo', original,
                     '--params-csv', args.params_csv.resolve(), '--limit', args.limit,
                     '--mode', 'fixed', '--output-dir', original_output], original, out/'logs/original.log')
            else:
                run([sys.executable, '-m', 'src.experiments.run_large_scale_rollout',
                     '--n-scenarios', args.n_scenarios, '--seed', args.seed, '--scenario-mix', 'decision_rich',
                     '--skip-state-labels', '--output-dir', original_output], original, out/'logs/original.log')
            if args.original_results:
                verify_historical(original_output/RESULTS_FILE, args.original_results.resolve())
                manifest['historical_score_match'] = True
            else:
                manifest['historical_score_match'] = 'not checked; this is not verification of old tables'
            run([sys.executable, TOOLS/'replay_saved_scenarios.py', '--repo', fixed,
                 '--params-csv', original_output/PARAMS_FILE, '--mode', 'fixed',
                 '--output-dir', fixed_output], fixed, out/'logs/fixed.log')
            summary = compare_results(original_output, fixed_output, out)
            print('\nNI in the code means NT in the proposal.\n')
            print(summary.to_string(index=False))
        manifest['status'] = 'completed'
        print('\nRESULTS:', out/'boundary_fix_comparison.csv', flush=True)
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = repr(exc)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
