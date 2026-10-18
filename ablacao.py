"""Task 3 — Ablação + Regressão logística e árvore de decisão.

Módulo AUTÓNOMO: constrói o seu próprio pré-processamento a partir das regras da
Task 1 (secção 6) e da Task 2 (secção 8.6), para permitir treinar com SUBCONJUNTOS de
grupos de variáveis (a ablação). Tudo o que aprende com os dados (medianas, limite,
média/desvio, categorias) é ajustado só no treino de cada fold, porque o pipeline é
reajustado dentro de `fazer_score`, chamada pelo `avaliar_folds`.

"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ── Dados ────────────────────────────────────────────────────────────────────
NORMAL_CAUSES = ["Charging ends normally", "User stops charging"]
VAL_END = pd.Timestamp("2021-10-01")          # o teste (out–dez/2021) nunca é carregado

NIVEIS_ESPACIAIS = {"posto": "Charging Post ID",
                    "localizacao": "Location Information",
                    "distrito": "District Name"}

# ── Grupos de variáveis (os mesmos da Parte 1, secção 8.2) ──────────────────
GRUPOS = {
    "tempo": ["hour", "dow"],
    "meteorologia": ["Temperature(℃)", "Relative Humidity(%)", "Precipitation(mm)"],
    "tarifa": ["electricity_price"],
    "historico_posto": [
        "post_previous_sessions_1D", "post_previous_abnormal_1D", "post_previous_abnormal_rate_1D",
        "post_previous_sessions_7D", "post_previous_abnormal_7D", "post_previous_abnormal_rate_7D",
        "post_previous_sessions_30D", "post_previous_abnormal_30D", "post_previous_abnormal_rate_30D",
        "post_hours_since_previous_completed_session", "post_hours_since_previous_completed_abnormal",
        "post_last_completed_abnormal"],
    "historico_utilizador": [
        "user_previous_sessions_30D", "user_previous_abnormal_30D", "user_previous_abnormal_rate_30D",
        "user_previous_sessions_180D", "user_previous_abnormal_180D", "user_previous_abnormal_rate_180D",
        "user_days_since_previous_completed_session", "user_days_since_previous_completed_abnormal",
        "user_no_previous_completed_session", "user_no_previous_completed_abnormal",
        "user_active_sessions_at_start", "user_last_completed_abnormal"],
}

# Regras de preparação (Task 1, secção 6)
IMPUTAR_MEDIANA = ["post_hours_since_previous_completed_session", "post_hours_since_previous_completed_abnormal",
                   "user_days_since_previous_completed_session", "user_days_since_previous_completed_abnormal"]
LIMITE_P99 = ["user_active_sessions_at_start"]
LOG1P = ["Precipitation(mm)",
         "post_previous_abnormal_1D", "post_previous_abnormal_7D", "post_previous_abnormal_30D",
         "post_hours_since_previous_completed_session", "post_hours_since_previous_completed_abnormal",
         "user_previous_sessions_30D", "user_previous_abnormal_30D",
         "user_previous_sessions_180D", "user_previous_abnormal_180D",
         "user_days_since_previous_completed_session", "user_days_since_previous_completed_abnormal"]
BINARIAS = ["user_no_previous_completed_session", "user_no_previous_completed_abnormal",
            "post_last_completed_abnormal", "user_last_completed_abnormal"]   # nunca escaladas


def carregar_dev(caminho):
    """Lê o CSV, deriva o alvo e as variáveis derivadas, e REMOVE o teste (out–dez/2021)."""
    df = pd.read_csv(caminho, parse_dates=["Start Time"])
    df.columns = df.columns.str.strip()
    df["end_cause"] = df["end_cause"].str.strip()
    df["is_Abnormal"] = (~df["end_cause"].isin(NORMAL_CAUSES)).astype(int)
    df = df[df["Start Time"] < VAL_END].sort_values("Start Time", kind="stable").reset_index(drop=True)
    df["hour"] = df["Start Time"].dt.hour
    df["dow"] = df["Start Time"].dt.dayofweek
    # a última sessão concluída foi anómala ⇔ tempo desde a anterior = tempo desde a anomalia anterior
    for ent, a, b in [("post", "post_hours_since_previous_completed_session", "post_hours_since_previous_completed_abnormal"),
                      ("user", "user_days_since_previous_completed_session", "user_days_since_previous_completed_abnormal")]:
        # mesma definição da Task 1 (np.isclose com tolerâncias por omissão; NaN → 0)
        df[f"{ent}_last_completed_abnormal"] = np.isclose(df[a], df[b]).astype(int)
    return df


# ── Pré-processamento ────────────────────────────────────────────────────────
class LimiteSuperior(BaseEstimator, TransformerMixin):
    """Limita cada coluna ao percentil 99 do TREINO."""
    def __init__(self, q=0.99):
        self.q = q
    def fit(self, X, y=None):
        self.limite_ = np.nanquantile(np.asarray(X, dtype=float), self.q, axis=0)
        return self
    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.limite_)
    def get_feature_names_out(self, names=None):
        return np.asarray(names)


def colunas_do_modelo(grupos, espaco="posto"):
    """Lista de colunas de entrada para um conjunto de grupos + nível espacial (ou None)."""
    cols = []
    if espaco is not None:
        cols.append(NIVEIS_ESPACIAIS[espaco])
    for g in grupos:
        cols += GRUPOS[g]
    return cols


def criar_preprocessador(cols, espaco="posto", escalar=True, hora_onehot=True):
    """ColumnTransformer só com as colunas pedidas. Ordem: imputar → limite → log1p → escalar."""
    cols = list(cols)
    esp_col = NIVEIS_ESPACIAIS[espaco] if espaco is not None else None
    cat = [esp_col] if esp_col in cols else []
    tempo_cat = [c for c in ("hour", "dow") if c in cols] if hora_onehot else []
    num = [c for c in cols if c not in cat and c not in tempo_cat]

    blocos = []
    # numéricas com imputação/limite/log1p aplicados nas colunas a que se aplicam
    num_imp = [c for c in num if c in IMPUTAR_MEDIANA]
    num_cap = [c for c in num if c in LIMITE_P99]
    num_log = [c for c in num if c in LOG1P]
    num_bin = [c for c in num if c in BINARIAS]
    num_resto = [c for c in num if c not in num_imp + num_cap + num_log + num_bin]

    def pipe_num(cs, imputar=False, limitar=False, log=False, escalar_=True):
        passos = []
        if imputar: passos.append(("imp", SimpleImputer(strategy="median")))
        if limitar: passos.append(("cap", LimiteSuperior()))
        if log: passos.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        if escalar and escalar_: passos.append(("esc", StandardScaler()))
        return Pipeline(passos) if passos else "passthrough"

    # colunas que são imputadas e (todas) também têm log1p
    imp_log = [c for c in num_imp if c in LOG1P]
    log_so = [c for c in num_log if c not in IMPUTAR_MEDIANA]
    if imp_log:   blocos.append(("imp_log", pipe_num(imp_log, imputar=True, log=True), imp_log))
    if log_so:    blocos.append(("log", pipe_num(log_so, log=True), log_so))
    if num_cap:   blocos.append(("cap", pipe_num(num_cap, limitar=True), num_cap))
    if num_resto: blocos.append(("resto", pipe_num(num_resto), num_resto))
    if num_bin:   blocos.append(("bin", "passthrough", num_bin))
    if cat:       blocos.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if tempo_cat:
        cats = [list(range(24)) if c == "hour" else list(range(7)) for c in tempo_cat]
        blocos.append(("tempo", OneHotEncoder(categories=cats, handle_unknown="ignore", sparse_output=False), tempo_cat))
    usadas = [c for _, _, cs in blocos for c in cs]
    assert sorted(usadas) == sorted(cols), f"colunas perdidas ou repetidas: {set(cols) ^ set(usadas)}"
    return ColumnTransformer(blocos, remainder="drop", verbose_feature_names_out=False)


# ── Modelos (hiperparâmetros FIXOS a priori; afinados só depois, na Parte B) ──
HIPERPARAMETROS = {
    "logistica": {"C": 0.1, "max_iter": 300},
    "arvore": {"max_depth": 8, "min_samples_leaf": 200},
}


def criar_modelo(nome, **hp):
    p = {**HIPERPARAMETROS[nome], **hp}
    if nome == "logistica":
        return LogisticRegression(random_state=0, **p)
    if nome == "arvore":
        return DecisionTreeClassifier(random_state=0, **p)
    raise ValueError(nome)


def criar_pipeline(nome, grupos, espaco="posto", **hp):
    """Pipeline completo: logística → hora em one-hot e escalonamento; árvore → hora numérica, sem escalonamento."""
    cols = colunas_do_modelo(grupos, espaco)
    linear = nome == "logistica"
    pre = criar_preprocessador(cols, espaco, escalar=linear, hora_onehot=linear)
    return cols, Pipeline([("pre", pre), ("modelo", criar_modelo(nome, **hp))])


def fabricar_scorer(nome, grupos, espaco="posto", treino_a_partir_de=None, **hp):
    """Devolve `fazer_score(treino, validacao)` para o `avaliar_folds` da Task 2.

    O pipeline é criado e ajustado de raiz em cada fold, só com o treino desse fold.
    `treino_a_partir_de`: data mínima do treino (para testar "sem o período de arranque").
    """
    def fazer_score(tr, va):
        if treino_a_partir_de is not None:
            tr = tr[tr["Start Time"] >= pd.Timestamp(treino_a_partir_de)]
        cols, pipe = criar_pipeline(nome, grupos, espaco, **hp)
        pipe.fit(tr[cols], tr["is_Abnormal"])
        return pipe.predict_proba(va[cols])[:, 1]
    return fazer_score


# ── Configurações da ablação (Task 3, bloco A) ───────────────────────────────
CONFIGURACOES = {
    "1 contexto + meteorologia": ["tempo", "meteorologia"],
    "2 + tarifa": ["tempo", "meteorologia", "tarifa"],
    "3 + histórico do posto": ["tempo", "meteorologia", "tarifa", "historico_posto"],
    "4 + histórico do utilizador": ["tempo", "meteorologia", "tarifa", "historico_posto", "historico_utilizador"],
}