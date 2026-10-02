#!/usr/bin/env python3
"""Regenerates RUN_QUEUE_20261001.md from the queue master log + per-job logs:
step states, latest progress line per job, and every result line so far."""
import glob, os, re, subprocess, datetime

BASE = '/raid/ruban/hpmlproj/term_project/SigSpatial'
LOGD = f'{BASE}/logs_20261001'
OUT = f'{BASE}/RUN_QUEUE_20261001.md'


def tail_match(path, pat, n=1):
    try:
        lines = [l.rstrip() for l in open(path, errors='ignore') if re.search(pat, l)]
        return lines[-n:]
    except FileNotFoundError:
        return []


steps = {}
master = f'{LOGD}/queue_master.log'
for l in open(master, errors='ignore') if os.path.exists(master) else []:
    m = re.match(r'\[(.*?)\] (START|DONE|FAILED|SKIPPED) (\S+)(.*)', l.strip())
    if m:
        steps[m.group(3)] = (m.group(2), m.group(1), m.group(4).strip())

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
df = subprocess.run(['df', '-h', '/raid'], capture_output=True, text=True).stdout.splitlines()[-1].split()
gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip().replace('\n', '; ')
L = [f'# Full-scale run queue (started 2026-10-01) -- status at {now}', '',
     'Queue: sports build (CPU, parallel) | GPU chain: ov1m train -> ov10m train -> sports train; '
     'each eval runs in the background as soon as its training finishes.', '',
     f'Disk /raid: {df[3]} free ({df[4]} used). GPUs: {gpu}', '',
     '| step | state | since | note |', '|---|---|---|---|']
order = ['build_sports', 'train_ov1m', 'eval_ov1m', 'train_ov10m', 'eval_ov10m', 'train_sportsfull', 'eval_sportsfull']
for s in order:
    st, t, note = steps.get(s, ('pending', '', ''))
    L.append(f'| {s} | {st} | {t} | {note} |')
L += ['', '## Latest progress per job', '']
for s in order:
    p = f'{LOGD}/{s}.log'
    if os.path.exists(p):
        last = tail_match(p, r'(step |tile |encodings |GT |embedded |listwise_ce|DONE|base ALLq|Error|error|Traceback)')
        L.append(f'- **{s}**: `{last[0][:220] if last else "(no progress line yet)"}`')
L += ['', '## Results so far (Stage-1, no rerank)', '']
for s in ['eval_ov1m', 'eval_ov10m', 'eval_sportsfull']:
    for r in tail_match(f'{LOGD}/{s}.log', r'base ALLq', n=20):
        L.append(f'- `{r[:200]}`')
open(OUT, 'w').write('\n'.join(L) + '\n')
