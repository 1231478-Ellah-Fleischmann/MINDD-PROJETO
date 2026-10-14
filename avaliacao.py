"""Avaliação dos modelos — Task 2

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


def alertas_por_orcamento(score, alertas_por_1000, seed=0):
    """Alerta exatamente nas round(n·X/1000) sessões com score mais alto.

    Os empates no limite são resolvidos ao acaso. É necessário para scores com poucos
    valores distintos (ex.: regra binária), em que `score >= limiar` geraria muito mais
    ou muito menos alertas do que o orçamento.
    """
    score = np.asarray(score, dtype=float)
    n = len(score)
    k = int(round(n * alertas_por_1000 / 1000))
    desempate = np.random.default_rng(seed).random(n)
    ordem = np.lexsort((desempate, -score))     # 1.º score decrescente, 2.º desempate aleatório
    pred = np.zeros(n, dtype=int)
    pred[ordem[:k]] = 1
    return pred


def avaliar(y, score, limiar=None, pred=None):
    """Métricas de ranking (independentes do limiar) e de decisão (com limiar).

    Alerta = score >= limiar. Em alternativa, os alertas podem ser dados já prontos em `pred`.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if pred is None:
        if limiar is None:
            raise ValueError("Indicar `limiar` ou `pred`.")
        pred = (score >= limiar).astype(int)
    else:
        pred = np.asarray(pred).astype(int)
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


# ── Validação temporal (Parte 2) ─────────────────────────────────────────────
# Expanding window: o treino começa sempre no início dos dados e cresce;
# a validação são os 3 meses seguintes. O teste (out–dez/2021) nunca entra aqui.
FOLDS = [
    # nome, início da validação, fim da validação (exclusivo)
    ("F1", "2020-10-01", "2021-01-01"),
    ("F2", "2021-01-01", "2021-04-01"),
    ("F3", "2021-04-01", "2021-07-01"),
    ("F4", "2021-07-01", "2021-10-01"),
]


def folds_temporais(datas, folds=FOLDS):
    """Gera (nome, idx_treino, idx_validacao) com índices POSICIONAIS.

    Treino = todas as sessões com início antes do início da validação.
    """
    datas = pd.to_datetime(pd.Series(datas)).reset_index(drop=True)
    for nome, ini_val, fim_val in folds:
        ini, fim = pd.Timestamp(ini_val), pd.Timestamp(fim_val)
        idx_tr = np.flatnonzero(datas < ini)
        idx_va = np.flatnonzero((datas >= ini) & (datas < fim))
        yield nome, idx_tr, idx_va


def avaliar_folds(df, fazer_score, alertas_por_1000=200, limiar=None,
                  col_y="is_Abnormal", col_t="Start Time", seed=0):
    """Corre um método de scoring nos 4 folds e devolve as métricas por fold.

    fazer_score(treino, validacao) -> scores para as sessões da validação.
    A função só pode aprender com `treino` (o treino do fold).

    Decisão:
      - limiar=None -> orçamento: alerta nas `alertas_por_1000` sessões de maior score
                       de cada validação (empates resolvidos ao acaso);
      - limiar=valor -> regra fixa: alerta se score >= limiar.
    """
    df = df.reset_index(drop=True)
    linhas = {}
    for nome, tr, va in folds_temporais(df[col_t]):
        d_tr, d_va = df.iloc[tr], df.iloc[va]
        s = np.asarray(fazer_score(d_tr, d_va), dtype=float)
        assert len(s) == len(d_va), "fazer_score tem de devolver um score por sessão de validação"
        y = d_va[col_y].to_numpy()
        if limiar is None:
            linhas[nome] = avaliar(y, s, pred=alertas_por_orcamento(s, alertas_por_1000, seed))
        else:
            linhas[nome] = avaliar(y, s, limiar=limiar)
    return pd.DataFrame(linhas).T.astype(float)


def resumir_folds(res, metricas=("pr_auc", "pr_auc_lift", "roc_auc", "recall", "precisao")):
    """Média, desvio-padrão e pior fold de cada métrica (todas: quanto maior, melhor)."""
    r = res[list(metricas)].astype(float)
    return pd.concat({"média": r.mean(), "desvio": r.std(ddof=1), "pior fold": r.min()}, axis=1).stack()
