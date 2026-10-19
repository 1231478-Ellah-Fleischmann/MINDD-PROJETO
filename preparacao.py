"""Preparação dos dados — Task 2, Parte 2

Passa a preparação da Task 1 para uma Pipeline do scikit-learn, para que
TUDO o que aprende valores dos dados (mediana da imputação, cap no p99,
média/desvio do scaler, categorias do one-hot) seja reajustado em cada fold,
só com o treino desse fold.

As escolhas de variáveis (quais entram, quais levam log1p, quais são binárias)
são FIXAS: foram decididas na Task 1 e não dependem do fold.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NORMAL_CAUSES = ["Charging ends normally", "User stops charging"]

# ── Preditores (33), iguais aos da Task 1 / Parte 1 ──────────────────────────
METEO = ["Temperature(℃)", "Relative Humidity(%)", "Precipitation(mm)"]
TARIFA = ["electricity_price"]
TEMPO = ["hour", "dow"]
HIST_POSTO = [
    "post_previous_sessions_1D", "post_previous_abnormal_1D", "post_previous_abnormal_rate_1D",
    "post_previous_sessions_7D", "post_previous_abnormal_7D", "post_previous_abnormal_rate_7D",
    "post_previous_sessions_30D", "post_previous_abnormal_30D", "post_previous_abnormal_rate_30D",
    "post_hours_since_previous_completed_session", "post_hours_since_previous_completed_abnormal",
    "post_last_completed_abnormal",
]
HIST_USER = [
    "user_previous_sessions_30D", "user_previous_abnormal_30D", "user_previous_abnormal_rate_30D",
    "user_previous_sessions_180D", "user_previous_abnormal_180D", "user_previous_abnormal_rate_180D",
    "user_days_since_previous_completed_session", "user_no_previous_completed_session",
    "user_days_since_previous_completed_abnormal", "user_no_previous_completed_abnormal",
    "user_active_sessions_at_start", "user_last_completed_abnormal",
]
LOCAL = ["Charging Post ID", "Location Information", "District Name"]

NUM = METEO + TARIFA + TEMPO + HIST_POSTO + HIST_USER      # 30
CAT = LOCAL                                                # 3
PREDITORES = NUM + CAT                                     # 33

GRUPOS = {"meteorologia": METEO, "tarifa": TARIFA, "tempo": TEMPO,
          "histórico do posto": HIST_POSTO, "histórico do utilizador": HIST_USER,
          "localização": LOCAL}

# ── Decisões fixas da Task 1 (secções 6.2, 6.3, 8.1 e 9.1-9.4) ─────────────────────────────
# Com valores em falta estruturais (imputados pela mediana; todas levam também log1p)
IMPUTAR_LOG = ["post_hours_since_previous_completed_session", "post_hours_since_previous_completed_abnormal",
               "user_days_since_previous_completed_session", "user_days_since_previous_completed_abnormal"]
# Restantes variáveis com assimetria > 3 no treino da Task 1 (só log1p)
LOG = ["Precipitation(mm)",
       "post_previous_abnormal_1D", "post_previous_abnormal_7D", "post_previous_abnormal_30D",
       "user_previous_sessions_30D", "user_previous_abnormal_30D",
       "user_previous_sessions_180D", "user_previous_abnormal_180D"]
CAP = ["user_active_sessions_at_start"]                    # limite no p99 do treino
BIN = ["user_no_previous_completed_session", "user_no_previous_completed_abnormal",
       "post_last_completed_abnormal", "user_last_completed_abnormal"]   # 0/1, sem escalonamento
OUTRAS = [c for c in NUM if c not in IMPUTAR_LOG + LOG + CAP + BIN]     # só escalonamento

assert sorted(IMPUTAR_LOG + LOG + CAP + BIN + OUTRAS) == sorted(NUM)


def carregar_dados(caminho, ate=None):
    """Lê só as colunas necessárias, deriva o alvo e as variáveis derivadas.

    `ate`: se indicado, só ficam as sessões com Start Time < ate (para deixar o teste de fora).
    """
    derivadas = set(TEMPO) | {"post_last_completed_abnormal", "user_last_completed_abnormal"}
    precisas = {c for c in PREDITORES if c not in derivadas} | {
        "Start Time", "end_cause", "UserID",
        "post_no_previous_completed_session"}  # só para verificação
    df = pd.read_csv(caminho, usecols=lambda c: c.strip() in precisas)
    df.columns = df.columns.str.strip()
    df["Start Time"] = pd.to_datetime(df["Start Time"])
    if ate is not None:
        df = df[df["Start Time"] < pd.Timestamp(ate)]
    df["end_cause"] = df["end_cause"].str.strip()
    df["is_Abnormal"] = (~df["end_cause"].isin(NORMAL_CAUSES)).astype(int)
    df = adicionar_derivadas(df)
    return df.sort_values("Start Time").reset_index(drop=True)


def adicionar_derivadas(df):
    """Variáveis derivadas da Task 1. São fórmulas fixas: não aprendem nada dos dados."""
    df = df.copy()
    df["hour"] = df["Start Time"].dt.hour
    df["dow"] = df["Start Time"].dt.dayofweek
    # a última sessão concluída foi anómala? (tempo desde a sessão anterior == tempo desde a anomalia anterior)
    df["post_last_completed_abnormal"] = np.isclose(df["post_hours_since_previous_completed_session"],
                                                    df["post_hours_since_previous_completed_abnormal"]).astype(int)
    df["user_last_completed_abnormal"] = np.isclose(df["user_days_since_previous_completed_session"],
                                                    df["user_days_since_previous_completed_abnormal"]).astype(int)
    return df


class CapQuantil(BaseEstimator, TransformerMixin):
    """Limita cada coluna ao quantil `q` aprendido no fit (no treino do fold)."""

    def __init__(self, q=0.99):
        self.q = q

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.limites_ = np.nanquantile(np.asarray(X, dtype=float), self.q, axis=0)
        self.n_features_in_ = len(self.limites_)
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.limites_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features, dtype=object)


def criar_preprocessador(escalar=True):
    """Pré-processamento da Task 1 como ColumnTransformer.

    escalar=True  -> versão para modelos lineares/distâncias (com StandardScaler)
    escalar=False -> versão para árvores (sem escalonamento; o resto é igual)
    """
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    esc = StandardScaler() if escalar else "passthrough"

    pre = ColumnTransformer(
        transformers=[
            ("imputar_log", Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("log", log1p), ("esc", esc)]), IMPUTAR_LOG),
            ("log", Pipeline([("log", log1p), ("esc", esc)]), LOG),
            ("cap", Pipeline([("cap", CapQuantil(0.99)), ("esc", esc)]), CAP),
            ("outras", Pipeline([("esc", esc)]), OUTRAS),
            ("bin", "passthrough", BIN),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")


def parametros_aprendidos(pre):
    """Resumo dos valores aprendidos por um pré-processador já ajustado (para o relatório)."""
    t = pre.named_transformers_
    out = {f"mediana {c}": v for c, v in zip(IMPUTAR_LOG, t["imputar_log"].named_steps["imp"].statistics_)}
    out["cap p99 user_active_sessions_at_start"] = t["cap"].named_steps["cap"].limites_[0]
    esc = t["outras"].named_steps["esc"]
    if esc != "passthrough":
        for c in ["post_previous_sessions_7D", "Temperature(℃)"]:
            i = OUTRAS.index(c)
            out[f"média {c}"] = esc.mean_[i]
            out[f"desvio {c}"] = esc.scale_[i]
    out["categorias Charging Post ID"] = len(t["cat"].categories_[0])
    return out
