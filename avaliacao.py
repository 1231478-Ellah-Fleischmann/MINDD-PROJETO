"""Avaliação dos modelos — Task 2, Parte 1.

Funções partilhadas pela Parte 1 (métricas) e pela Parte 2 (folds e baselines).
Convenção: `score` = quanto maior, maior o risco de sessão anómala (classe 1).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score, recall_score, fbeta_score)


def limiar_por_orcamento(score, alertas_por_1000):
    """Limiar de score que gera ~`alertas_por_1000` alertas por 1000 sessões."""
    q = 1 - alertas_por_1000 / 1000
    return float(np.quantile(np.asarray(score, dtype=float), q))


def avaliar(y, score, limiar):
    """Métricas de ranking (independentes do limiar) e de decisão (com limiar).

    Alerta = score >= limiar.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= limiar).astype(int)
    n = len(y)
    prev = y.mean()

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    # PR-AUC e ROC-AUC só existem se houver as duas classes (ex.: meses pequenos)
    if 0 < y.sum() < n:
        pr_auc = average_precision_score(y, score)
        roc_auc = roc_auc_score(y, score)
    else:
        pr_auc = roc_auc = np.nan

    return {
        "n": n,
        "prevalencia": prev,
        # ranking
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / prev if prev > 0 else np.nan,
        "roc_auc": roc_auc,
        # decisão
        "recall": recall_score(y, pred, zero_division=0),
        "precisao": precision_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        # operacionais (por 1000 sessões)
        "alertas_corretos_1000": 1000 * tp / n,
        "falsos_alertas_1000": 1000 * fp / n,
        "falhadas_1000": 1000 * fn / n,
        "total_alertas_1000": 1000 * (tp + fp) / n,
    }


def avaliar_por_mes(datas, y, score, limiar):
    """`avaliar` aplicado a cada mês (para a medida de estabilidade temporal)."""
    d = pd.DataFrame({"mes": pd.to_datetime(datas).dt.to_period("M"),
                      "y": np.asarray(y), "score": np.asarray(score, dtype=float)})
    linhas = {mes: avaliar(g["y"], g["score"], limiar) for mes, g in d.groupby("mes")}
    return pd.DataFrame(linhas).T