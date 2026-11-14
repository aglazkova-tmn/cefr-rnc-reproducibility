"""CSV input/output shared by the final CEFR model notebook."""
from pathlib import Path
import numpy as np
import pandas as pd


def read_new_texts(path, sep=','):
    # Preserve original values, including strings such as NA, IDs and leading zeros.
    frame = pd.read_csv(path, sep=sep, encoding='utf-8-sig',
                        dtype=str, keep_default_na=False)
    if 'text' not in frame:
        raise ValueError('CSV должен содержать столбец text')
    if frame.empty:
        raise ValueError('CSV не содержит строк')
    invalid = ~frame.text.map(lambda text: bool(text.strip()))
    if invalid.any():
        rows = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()
        raise ValueError(f'Пустой text в строках CSV (с учётом заголовка): {rows[:30]}')
    return frame


def rounded_levels(raw):
    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError('Получен нечисловой прогноз')
    return np.clip(np.floor(raw + .5), 1, 6).astype(int)


def save_predictions(frame, predictions, path):
    """Preserve input order and columns; refuse accidental column overwrites."""
    collisions = set(frame.columns) & set(predictions)
    if collisions:
        raise ValueError(f'В исходном CSV уже есть столбцы результатов: {sorted(collisions)}')
    result = frame.copy()
    for name, values in predictions.items():
        values = np.asarray(values)
        if values.shape != (len(frame),):
            raise ValueError(f'Неверное число прогнозов в {name}')
        if not np.isfinite(values).all():
            raise ValueError(f'Нечисловые значения в {name}')
        result[name] = values
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding='utf-8-sig')
    return result


def dual_predictions(rubert_raw, plain_probabilities, weighted_probabilities):
    """Both ensembles use exactly the same RuBERT vector; MLP classes use argmax."""
    raw=np.asarray(rubert_raw,dtype=float)
    if raw.ndim!=1 or not np.isfinite(raw).all():raise ValueError('Invalid RuBERT vector')
    result={'rubert_raw':raw,'rubert_rounded':rounded_levels(raw)}
    for name,values in [('mlp',plain_probabilities),('mlp_weighted',weighted_probabilities)]:
        probs=np.asarray(values,dtype=float)
        if probs.shape!=(len(raw),6) or not np.isfinite(probs).all():raise ValueError('Invalid probability shape/values')
        if (probs<0).any() or (probs>1).any() or not np.allclose(probs.sum(1),1,atol=1e-6,rtol=0):
            raise ValueError('Invalid probabilities')
        for k in range(6):result[f'{name}_probability_{k+1}']=probs[:,k]
        expected=probs@np.arange(1,7)
        result[f'{name}_class']=probs.argmax(1)+1
        result[f'{name}_expected_level']=expected
        ensemble=(raw+expected)/2
        prefix='ensemble' if name=='mlp' else 'ensemble_weighted_mlp'
        result[f'{prefix}_raw']=ensemble
        result[f'{prefix}_rounded']=rounded_levels(ensemble)
    return result
