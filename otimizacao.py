"""Task 4 — Parte A: otimização de hiperparâmetros com folds temporais.

Funções partilhadas pelo notebook `task4_otimizacao.ipynb`. Tudo o que aprende com os dados
(medianas, limite, escalas, categorias, o próprio modelo) é reajustado em cada fold só com o treino
desse fold, porque o ajuste passa pelo `avaliar_folds` da Task 2 e pelo `fabricar_scorer` da Task 3.
O teste (out–dez/2021) nunca é carregado (ver `ablacao.carregar_dev`).

Regra de escolha (secção 8.9 do relatório): ordenar pelo recall médio com 200 alertas por 1000
sessões; diferenças inferiores a 0,005 são empate, desempatado pela PR-AUC normalizada. Se a
configuração inicial está no empate e o desempate não a supera em mais de 0,005 de PR-AUC, mantém-se
a inicial (parcimónia: não trocar por uma diferença que é ruído).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler

import ablacao as ab
from avaliacao import avaliar_folds, comparar, folds_temporais

TOLERANCIA = 0.005      # diferença de recall abaixo da qual dois candidatos empatam (secção 8.9)


# ── Candidatos ───────────────────────────────────────────────────────────────
def candidatos(espaco, n, seed=0):
    """Lista de configurações: a grelha inteira se tiver até `n` pontos, senão `n` sorteadas (aleatório, com semente)."""
    total = len(ParameterGrid(espaco))
    if total <= n:
        return list(ParameterGrid(espaco))
    return list(ParameterSampler(espaco, n_iter=n, random_state=seed))


def _chave(hp):
    return json.dumps(hp, sort_keys=True, default=str)


# ── Pesquisa ─────────────────────────────────────────────────────────────────
def pesquisar(df, modelo, lista_hp, grupos, espaco_geo, alertas=200, fixos=None,
              inicial=None, ficheiro=None, verbose=True):
    """Avalia cada configuração nos 4 folds temporais e devolve uma tabela (uma linha por configuração).

    - `fixos`: hiperparâmetros acrescentados a todas as configurações só durante a pesquisa
      (por exemplo, menos árvores na Random Forest, para poupar tempo).
    - `inicial`: a configuração de partida; é avaliada nas mesmas condições e entra na tabela
      com `inicial=True`, para comparar sempre como com como.
    - `ficheiro`: CSV de ponto de controlo. Se a pesquisa for interrompida, volta a correr-se
      e só se calculam as configurações que faltam.
    """
    fixos = fixos or {}
    todos = ([(inicial, True)] if inicial is not None else []) + [(hp, False) for hp in lista_hp]
    vistos, unicos = set(), []          # uma configuração repetida (por exemplo, a inicial dentro da grelha) conta uma só vez
    for hp, e_ini in todos:
        if _chave(hp) not in vistos:
            vistos.add(_chave(hp)); unicos.append((hp, e_ini))
    todos = unicos
    feitos = {}
    if ficheiro and Path(ficheiro).exists():
        antigo = pd.read_csv(ficheiro)
        feitos = {r["chave"]: r.to_dict() for _, r in antigo.iterrows()}
    linhas = []
    for i, (hp, e_inicial) in enumerate(todos, 1):
        k = _chave([hp, fixos])          # a chave inclui os `fixos`: se mudarem, a configuração é recalculada
        if k in feitos:
            linha = feitos[k]; linha["inicial"] = e_inicial
        else:
            t0 = time.perf_counter()
            res = avaliar_folds(df, ab.fabricar_scorer(modelo, grupos, espaco_geo, **{**hp, **fixos}), alertas)
            linha = {"chave": k, "inicial": e_inicial, "hp": _chave(hp)}
            for fold, r in res.iterrows():
                linha[f"recall_{fold}"] = r["recall"]
                linha[f"prauc_{fold}"] = r["pr_auc_norm"]
            linha["recall_media"] = res["recall"].mean()
            linha["recall_desvio"] = res["recall"].std(ddof=1)
            linha["recall_pior"] = res["recall"].min()
            linha["prauc_media"] = res["pr_auc_norm"].mean()
            linha["roc_media"] = res["roc_auc"].mean()
            linha["tempo_s"] = time.perf_counter() - t0
            if verbose:
                print(f"[{modelo}] {i}/{len(todos)}  recall {linha['recall_media']:.4f}  "
                      f"PR-AUC norm. {linha['prauc_media']:.3f}  ({linha['tempo_s']:.0f} s)  {_chave(hp)}")
        linhas.append(linha)
        if ficheiro:
            pd.DataFrame(linhas).to_csv(ficheiro, index=False)
    return pd.DataFrame(linhas)


def folds_da_tabela(tabela):
    return [c[len("recall_"):] for c in tabela.columns if c.startswith("recall_F")]


# ── Escolha (regra da secção 8.9) ────────────────────────────────────────────
def escolher(tabela, tol=TOLERANCIA, folds=None):
    """Melhor configuração: dentro de `tol` do melhor recall médio, a de maior PR-AUC normalizada.

    `folds`: lista de folds a usar (por omissão, todos). Serve para a verificação em que se
    escolhe só com F1–F3 e se avalia em F4.
    """
    t = tabela.copy()
    fs = folds or folds_da_tabela(t)
    t["_recall"] = t[[f"recall_{f}" for f in fs]].mean(axis=1)
    t["_prauc"] = t[[f"prauc_{f}" for f in fs]].mean(axis=1)
    empate = t[t["_recall"] >= t["_recall"].max() - tol]
    melhor = empate.sort_values("_prauc", ascending=False).iloc[0]
    # Parcimónia: se a inicial está no empate e o desempate não a supera em mais de `tol` de PR-AUC
    # normalizada, a diferença é ruído e mantém-se a inicial.
    ini = empate[empate["inicial"].astype(bool)]
    if len(ini) and melhor["_prauc"] - ini.iloc[0]["_prauc"] <= tol:
        return ini.iloc[0]
    return melhor


def serie_recall(linha, folds=None):
    fs = folds or folds_da_tabela(linha.to_frame().T)
    return pd.DataFrame({"recall": [float(linha[f"recall_{f}"]) for f in fs]}, index=fs)


def veredicto(res_a, res_b):
    """Regra da secção 8.9 de A (otimizado) contra B (inicial): maior média, melhor em 3 de 4 folds, pior fold não inferior."""
    c = comparar(res_a, res_b, "recall")
    ok = bool(c["diferença média"] > 0 and c["folds em que A ganha"] >= 3 and c["pior fold A"] >= c["pior fold B"])
    return {**c, "melhoria real e estável (regra 8.9)": ok}


# ── Métricas das probabilidades ──────────────────────────────────────────────
def metricas_probabilisticas(scores, bins=10):
    """Brier, log-loss e erro de calibração esperado (ECE, bins de igual frequência), média dos folds.

    `scores`: dicionário {fold: (y, score, pred)} devolvido por `avaliar_folds(devolver_scores=True)`.
    """
    brier, ll, ece = [], [], []
    for y, s, _ in scores.values():
        y = np.asarray(y, dtype=float); s = np.clip(np.asarray(s, dtype=float), 1e-6, 1 - 1e-6)
        brier.append(np.mean((s - y) ** 2))
        ll.append(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))
        q = np.quantile(s, np.linspace(0, 1, bins + 1)); q[0] -= 1e-9; q[-1] += 1e-9
        b = np.digitize(s, q[1:-1])
        ece.append(sum((b == k).mean() * abs(y[b == k].mean() - s[b == k].mean()) for k in range(bins) if (b == k).any()))
    return {"Brier": np.mean(brier), "log-loss": np.mean(ll), "ECE": np.mean(ece)}


# ── Ficheiros para a Parte B ─────────────────────────────────────────────────
def previsoes_oof(df, scores):
    """Previsões de validação dos 4 folds (cada fold previsto por um modelo treinado só com dados anteriores)."""
    partes = []
    df = df.reset_index(drop=True)
    for nome, _, va in folds_temporais(df["Start Time"]):
        y, s, _pred = scores[nome]
        d = df.iloc[va]
        assert len(d) == len(s) and np.array_equal(d["is_Abnormal"].to_numpy(), np.asarray(y))
        partes.append(pd.DataFrame({"fold": nome, "linha_df": va, "Start Time": d["Start Time"].to_numpy(),
                                    "y": np.asarray(y).astype(int), "score": np.asarray(s, dtype=float)}))
    return pd.concat(partes, ignore_index=True)


def guardar_para_B(pasta, df, escolhidos, scores_por_modelo, modelo_probabilistico, extra=None):
    """Grava os hiperparâmetros escolhidos (JSON) e as previsões de validação de cada modelo (CSV comprimido)."""
    pasta = Path(pasta); pasta.mkdir(parents=True, exist_ok=True)
    for nome, sc in scores_por_modelo.items():
        previsoes_oof(df, sc).to_csv(pasta / f"task4_oof_validacao_{nome}.csv.gz", index=False)
    conteudo = {
        "modelo_probabilistico_mais_forte": modelo_probabilistico,
        "hiperparametros_escolhidos": escolhidos,
        "grupos_de_variaveis": "combinação 4 (tempo, meteorologia, tarifa, histórico do posto e do utilizador), sem informação espacial",
        "colunas_das_previsoes": {"fold": "F1–F4", "linha_df": "posição da sessão no df de ablacao.carregar_dev",
                                  "Start Time": "início da sessão", "y": "is_Abnormal", "score": "probabilidade prevista"},
        "nota": "Cada fold é previsto por um modelo ajustado só com o treino desse fold (janela crescente). "
                "O teste não foi usado.",
    }
    if extra:
        conteudo.update(extra)
    with open(pasta / "task4_modelos_escolhidos.json", "w", encoding="utf-8") as f:
        json.dump(conteudo, f, ensure_ascii=False, indent=2, default=str)