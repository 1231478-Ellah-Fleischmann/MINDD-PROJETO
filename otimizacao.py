"""Task 4 — Parte A: otimização de hiperparâmetros com folds temporais.

Funções partilhadas pelo notebook `task4_otimizacao.ipynb`. Tudo o que aprende com os dados
(medianas, limite, escalas, categorias, o próprio modelo) é reajustado em cada fold só com o treino
desse fold, porque o ajuste passa pelo `avaliar_folds` da Task 2 e pelo `fabricar_scorer` da Task 3.
O teste (out–dez/2021) nunca é carregado (ver `ablacao.carregar_dev`).

Regras (secção 8.9 do relatório, fixadas antes da otimização):
- Escolha de hiperparâmetros (`escolher`): ordenar pelo recall médio com 200 alertas por 1000
  sessões; diferenças inferiores a 0,005 são empate, desempatado pela PR-AUC normalizada.
- Comparação final (`veredicto`, `decidir`): A é preferido a B se tiver maior recall médio, for
  melhor em pelo menos 3 de 4 folds e tiver pior fold não inferior; senão é empate, resolvido pelo
  menor desvio-padrão entre folds, pela maior PR-AUC normalizada média e pela simplicidade.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler

import ablacao as ab
from avaliacao import FOLDS, avaliar_folds, comparar, folds_temporais

TOLERANCIA = 0.005      # diferença de recall abaixo da qual dois candidatos empatam (secção 8.9)
FOLDS_NOMES = [f[0] for f in FOLDS]   # ["F1", "F2", "F3", "F4"]


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
    return empate.sort_values("_prauc", ascending=False).iloc[0]


def serie_recall(linha, folds=None):
    fs = folds or folds_da_tabela(linha.to_frame().T)
    return pd.DataFrame({"recall": [float(linha[f"recall_{f}"]) for f in fs]}, index=fs)


def veredicto(res_a, res_b):
    """Regra da secção 8.9 de A (otimizado) contra B (inicial): maior média, melhor em 3 de 4 folds, pior fold não inferior."""
    c = comparar(res_a, res_b, "recall")
    ok = bool(c["diferença média"] > 0 and c["folds em que A ganha"] >= 3 and c["pior fold A"] >= c["pior fold B"])
    return {**c, "melhoria real e estável (regra 8.9)": ok}

def decidir(resultados, simplicidade):
    """Regra completa da secção 8.9 entre vários candidatos.

    `resultados`: {nome: tabela por fold com 'recall' e 'pr_auc_norm'} (de `avaliar_folds`).
    `simplicidade`: lista de nomes do mais simples para o mais complexo (último desempate).
    1. O candidato com maior recall médio ganha se for preferido (veredicto) a todos os outros.
    2. Senão, ele e os candidatos a que não é preferido ficam empatados, e escolhe-se, por esta
       ordem, o de menor desvio-padrão do recall, o de maior PR-AUC normalizada média e o mais simples.
    Devolve (nome escolhido, tabela com as comparações).
    """
    nomes = sorted(resultados, key=lambda n: resultados[n]["recall"].mean(), reverse=True)
    lider, outros = nomes[0], nomes[1:]
    linhas = {f"{lider} contra {o}": veredicto(resultados[lider], resultados[o]) for o in outros}
    tabela = pd.DataFrame(linhas).T
    empatados = [lider] + [o for o in outros
                           if not linhas[f"{lider} contra {o}"]["melhoria real e estável (regra 8.9)"]]
    if len(empatados) == 1:
        return lider, tabela
    chave = lambda n: (resultados[n]["recall"].std(ddof=1),       # 1.º menor desvio
                       -resultados[n]["pr_auc_norm"].mean(),       # 2.º maior PR-AUC normalizada
                       simplicidade.index(n))                      # 3.º mais simples
    return sorted(empatados, key=chave)[0], tabela

def elegivel(res, recall_ref, prauc_ref):
    """Condição de elegibilidade da secção 8.9, fold a fold.

    `recall_ref`: recall com 200 alertas da regra da última sessão anómala, por fold.
    `prauc_ref`: PR-AUC normalizada da taxa do posto a 7 dias, por fold.
    """
    t = pd.DataFrame({"recall": res["recall"], "recall ref.": pd.Series(recall_ref),
                      "PR-AUC norm.": res["pr_auc_norm"], "PR-AUC norm. ref.": pd.Series(prauc_ref)})
    t["supera as duas"] = (t["recall"] > t["recall ref."]) & (t["PR-AUC norm."] > t["PR-AUC norm. ref."])
    return t

# ── Scores de aquecimento (limiar móvel na primeira semana de cada fold) ─────
def com_aquecimento(fazer_score, dias=7):
    """Envolve um `fazer_score` para guardar também os scores dos `dias` antes de cada validação.

    O limiar móvel (secção 8.8) calcula o limiar da primeira semana com os scores dos 7 dias
    anteriores. Esses dias pertencem ao treino do fold e são previstos pelo modelo desse fold,
    tal como no teste a última semana de setembro é prevista pelo modelo final. O modelo é ajustado
    uma só vez, e os scores da validação ficam iguais aos que se obteriam sem o aquecimento.
    Devolve (novo fazer_score, lista que recebe um DataFrame por fold, pela ordem dos folds).
    """
    guardados = []

    def f(tr, va):
        inicio = va["Start Time"].min().normalize()          # os folds começam à meia-noite do dia 1
        aq = tr[tr["Start Time"] >= inicio - pd.Timedelta(days=dias)]
        s = np.asarray(fazer_score(tr, pd.concat([aq, va])), dtype=float)
        guardados.append(pd.DataFrame({"linha_df": aq.index.to_numpy(),
                                       "Start Time": aq["Start Time"].to_numpy(),
                                       "y": aq["is_Abnormal"].to_numpy().astype(int),
                                       "score": s[:len(aq)]}))
        return s[len(aq):]
    return f, guardados


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


def guardar_para_B(pasta, df, escolhidos, scores_por_modelo, modelo_probabilistico, extra=None,
                   aquecimento_por_modelo=None):
    """Grava os hiperparâmetros escolhidos (JSON), as previsões de validação de cada modelo e, se
    dados, os scores de aquecimento (7 dias antes de cada validação, previstos pelo modelo do fold)."""
    pasta = Path(pasta); pasta.mkdir(parents=True, exist_ok=True)
    for nome, sc in scores_por_modelo.items():
        previsoes_oof(df, sc).to_csv(pasta / f"task4_oof_validacao_{nome}.csv.gz", index=False)
    for nome, partes in (aquecimento_por_modelo or {}).items():
        assert len(partes) == len(FOLDS_NOMES), "tem de haver um bloco de aquecimento por fold"
        pd.concat([p.assign(fold=f) for f, p in zip(FOLDS_NOMES, partes)], ignore_index=True) \
            [["fold", "linha_df", "Start Time", "y", "score"]] \
            .to_csv(pasta / f"task4_aquecimento_{nome}.csv.gz", index=False)
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