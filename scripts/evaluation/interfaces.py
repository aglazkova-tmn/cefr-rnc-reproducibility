"""New packaging interfaces around the archived metric implementation.

These validation/CSV helpers are not historical experimental code.
"""
import csv
from pathlib import Path
import numpy as np
from .original_metrics import evaluate

LEVELS = ('A1', 'A2', 'B1', 'B2', 'C1', 'C2')

def label(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Boolean is not a CEFR label')
    if isinstance(value, str) and value.strip().upper() in LEVELS:
        return LEVELS.index(value.strip().upper()) + 1
    number = float(value)
    if not np.isfinite(number) or number not in range(1, 7):
        raise ValueError('Expected CEFR label A1–C2 or integer 1–6')
    return int(number)

def rounded(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite continuous prediction')
    return np.clip(np.floor(values + .5), 1, 6).astype(int)

def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty CSV')
    return rows

def validate(rows, gold='median_rating_round', require_fold=False):
    required = {'row_id', 'text', gold} | ({'fold'} if require_fold else set())
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Missing required columns: ' + ', '.join(sorted(required)))
    ids = [r['row_id'] for r in rows]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate row_id')
    if any(not isinstance(r['text'], str) or not r['text'].strip() for r in rows):
        raise ValueError('Empty or non-string text')
    y = np.array([label(r[gold]) for r in rows])
    if require_fold:
        f = [int(r['fold']) for r in rows]
        if set(f) != set(range(1, 6)) or any(str(v) != str(r['fold']) for v, r in zip(f, rows)):
            raise ValueError('Expected all five integer OOF fold labels 1–5')
    return y

def align(reference, extra, reference_gold='median_rating_round', extra_gold='target', require_fold=True):
    validate(reference, reference_gold, require_fold)
    validate(extra, extra_gold, require_fold)
    lookup = {r['row_id']: r for r in extra}
    if set(lookup) != {r['row_id'] for r in reference}:
        raise ValueError('Row ID sets differ')
    ordered = [lookup[r['row_id']] for r in reference]
    for left, right in zip(reference, ordered):
        if left['text'] != right['text'] or label(left[reference_gold]) != label(right[extra_gold]):
            raise ValueError('Text or gold label mismatch after ID join')
        if require_fold and left['fold'] != right['fold']:
            raise ValueError('OOF fold mismatch')
    return ordered

def metrics(y, prediction):
    y = np.array([label(v) for v in y]); p = np.array([label(v) for v in prediction])
    if not len(y) or y.shape != p.shape:
        raise ValueError('Nonempty aligned vectors required')
    cm, pr, re, f1, rf1, mae = evaluate(y, p)
    summary = {'Macro-F1': float(f1.mean()), 'Relaxed Macro-F1': float(rf1.mean()), 'MAE': mae}
    per_class = [{'Level': c, 'Precision': float(pr[i]), 'Recall': float(re[i]),
                  'F1': float(f1[i]), 'Support': int(cm[i].sum())} for i, c in enumerate(LEVELS)]
    for i in (0, 5):
        for k in ('Precision', 'Recall', 'F1'):
            summary[f'{LEVELS[i]} {k}'] = per_class[i][k]
    return summary, per_class, cm

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('No rows to write')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def save_report(directory, name, y, prediction, metadata=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    summary, per_class, cm = metrics(y, prediction)
    write_csv(directory / f'{name}_metrics.csv', [{'Model': name, **(metadata or {}), **summary}])
    write_csv(directory / f'{name}_per_class.csv', per_class)
    write_csv(directory / f'{name}_confusion_matrix.csv',
              [{'Gold': LEVELS[i], **dict(zip(LEVELS, map(int, row)))} for i, row in enumerate(cm)])
    return summary
