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
                  col_y="is_Abnormal", col_t="Start Time", seed=0, devolver_scores=False):
    """Corre um método de scoring nos 4 folds e devolve as métricas por fold.

    fazer_score(treino, validacao) -> scores para as sessões da validação.
    A função só pode aprender com `treino` (o treino do fold).

    Decisão:
      - limiar=None -> orçamento: alerta nas `alertas_por_1000` sessões de maior score
                       de cada validação (empates resolvidos ao acaso);
      - limiar=valor -> regra fixa: alerta se score >= limiar.

    devolver_scores=True -> devolve também {fold: (y, score, pred)}, necessário para o
    resultado pooled e para testar a transferência do limiar entre períodos.
    """
    df = df.reset_index(drop=True)
    linhas, scores = {}, {}
    for nome, tr, va in folds_temporais(df[col_t]):
        d_tr, d_va = df.iloc[tr], df.iloc[va]
        s = np.asarray(fazer_score(d_tr, d_va), dtype=float)
        assert len(s) == len(d_va), "fazer_score tem de devolver um score por sessão de validação"
        y = d_va[col_y].to_numpy()
        if limiar is None:
            pred = alertas_por_orcamento(s, alertas_por_1000, seed)
        else:
            pred = (s >= limiar).astype(int)
        linhas[nome] = avaliar(y, s, pred=pred)
        scores[nome] = (y, s, pred)
    res = pd.DataFrame(linhas).T.astype(float)
    return (res, scores) if devolver_scores else res


def resumir_folds(res, metricas=("pr_auc", "pr_auc_lift", "roc_auc", "recall", "precisao")):
    """Média, desvio-padrão e pior fold de cada métrica (todas: quanto maior, melhor)."""
    r = res[list(metricas)].astype(float)
    return pd.concat({"média": r.mean(), "desvio": r.std(ddof=1), "pior fold": r.min()}, axis=1).stack()



# ── Agregação dos folds (2.4) ────────────────────────────────────────────────
METRICAS_RESUMO = ("pr_auc", "pr_auc_lift", "roc_auc", "recall", "precisao")


def agregar_folds(res, scores=None, metricas=METRICAS_RESUMO):
    """Resumo de cada métrica entre folds.

    - média (sem ponderação): cada período conta o mesmo  -> agregação PRINCIPAL
    - desvio-padrão e pior fold: estabilidade
    - média ponderada pelo nº de sessões de validação
    - pooled: as 4 validações juntas como um só conjunto (se `scores` for dado)
    - tendência: declive da métrica por fold (F1 -> F4), por mínimos quadrados
    """
    r = res[list(metricas)].astype(float)
    w = res["n"].astype(float)
    out = pd.DataFrame({
        "média": r.mean(),
        "desvio": r.std(ddof=1),
        "pior fold": r.min(),
        "fold do pior": r.idxmin(),
        "média ponderada": r.mul(w, axis=0).sum() / w.sum(),
        "tendência por fold": [np.polyfit(np.arange(len(r)), r[m].to_numpy(), 1)[0] for m in r.columns],
    })
    if scores is not None:
        y = np.concatenate([v[0] for v in scores.values()])
        s = np.concatenate([v[1] for v in scores.values()])
        p = np.concatenate([v[2] for v in scores.values()])
        pooled = avaliar(y, s, pred=p)
        out["pooled"] = [pooled[m] for m in r.columns]
    return out


def comparar(res_a, res_b, metrica="pr_auc"):
    """Compara dois métodos fold a fold (diferenças emparelhadas A − B)."""
    a, b = res_a[metrica].astype(float), res_b[metrica].astype(float)
    d = a - b
    return {
        "diferença média": d.mean(),
        "desvio da diferença": d.std(ddof=1),
        "folds em que A ganha": int((d > 0).sum()),
        "pior fold A": a.min(),
        "pior fold B": b.min(),
    }


def transferir_limiar(scores, alertas_por_1000=200):
    """Testa se um limiar fixado num período gera o orçamento no período seguinte.

    Para cada fold k >= 2: limiar = quantil (1 − X/1000) dos scores da validação do fold k−1;
    aplica-se aos scores da validação do fold k, como se faria com o teste.
    """
    nomes = list(scores)
    linhas = {}
    for anterior, atual in zip(nomes[:-1], nomes[1:]):
        lim = limiar_por_orcamento(scores[anterior][1], alertas_por_1000)
        y, s, pred_orc = scores[atual]
        r = avaliar(y, s, limiar=lim)
        linhas[f"{anterior} → {atual}"] = {
            "limiar": lim,
            "alertas por 1000 (limiar transferido)": r["total_alertas_1000"],
            "recall (limiar transferido)": r["recall"],
            "precisão (limiar transferido)": r["precisao"],
            "recall (orçamento exato)": avaliar(y, s, pred=pred_orc)["recall"],
        }
    return pd.DataFrame(linhas).T
