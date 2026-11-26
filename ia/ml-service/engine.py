"""Motor de ML puro — carrega modelos e expõe feature engineering + inferência bruta."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

COMPONENTES = ["motor", "transmissao", "hidraulico", "freios"]
REGIMES = ["campo", "transporte", "ralenti", "parado"]
EQUIPAMENTOS = ["trator_novo", "trator_antigo", "trator_critico"]
# Demos reutilizam modelos do perfil correspondente.
EQUIPAMENTO_MODELO = {
    "demo_novo": "trator_novo",
    "demo_antigo": "trator_antigo",
    "demo_critico": "trator_critico",
}


def _modelo_para(equipamento_id: str) -> str:
    return EQUIPAMENTO_MODELO.get(equipamento_id, equipamento_id)


def engenharia_features(df: pd.DataFrame, base: List[str], window: int) -> pd.DataFrame:
    """Espelha o notebook: valor + roll_std + delta + regime one-hot."""
    feats = pd.DataFrame(index=df.index)
    for col in base:
        s = df[col].astype(float)
        feats[col] = s
        feats[f"{col}_roll_std"] = s.rolling(window, min_periods=1).std().fillna(0.0)
        feats[f"{col}_delta"] = s.diff().fillna(0.0)
    for r in REGIMES:
        feats[f"regime_{r}"] = (df["regime_operacao"] == r).astype(int)
    return feats.replace([np.inf, -np.inf], 0.0)


def features_streaming(
    df: pd.DataFrame,
    base: List[str],
    feature_cols: List[str],
    window: int = 6,
) -> pd.DataFrame:
    """Monta features linha a linha como a API: buffer de até `window` leituras."""
    rows: list[pd.Series] = []
    for i in range(len(df)):
        buffer = df.iloc[max(0, i - window + 1) : i + 1]
        X = engenharia_features(buffer, base, window)
        rows.append(X.iloc[-1])
    return pd.DataFrame(rows).reset_index(drop=True)[feature_cols]


class InferenceEngine:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        self.meta: Dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        for eq in EQUIPAMENTOS:
            eq_dir = self.models_dir / eq
            if not eq_dir.is_dir():
                raise FileNotFoundError(f"Pasta de modelos ausente: {eq_dir}")
            with open(eq_dir / "metadata.json", encoding="utf-8") as f:
                self.meta[eq] = json.load(f)
            self.pipelines[eq] = {}
            for comp in COMPONENTES:
                path = eq_dir / f"{comp}.joblib"
                if not path.exists():
                    raise FileNotFoundError(f"Modelo ausente: {path}")
                self.pipelines[eq][comp] = joblib.load(path)

    @property
    def equipamentos(self) -> List[str]:
        return list(self.pipelines.keys())

    def get_metadata(self, equipamento_id: str) -> dict:
        modelo = _modelo_para(equipamento_id)
        if modelo not in self.meta:
            raise ValueError(f"Equipamento desconhecido: {equipamento_id}")
        return self.meta[modelo]

    def buffer_to_dataframe(self, buffer: List[dict]) -> pd.DataFrame:
        df = pd.DataFrame(buffer)
        if "timestamp" in df.columns:
            # format="mixed" tolera timestamps com "T" e com espaço no mesmo
            # buffer; errors="coerce" evita quebra em valor malformado isolado.
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], format="mixed", errors="coerce"
            )
            df = df.sort_values("timestamp", na_position="last").reset_index(drop=True)
        return df.reset_index(drop=True)

    def raw_scores(self, equipamento_id: str, buffer: List[dict]) -> tuple[pd.DataFrame, Dict[str, float]]:
        modelo = _modelo_para(equipamento_id)
        if modelo not in self.pipelines:
            raise ValueError(f"Equipamento desconhecido: {equipamento_id}")
        if not buffer:
            raise ValueError("Buffer vazio — envie ao menos 1 leitura")

        df = self.buffer_to_dataframe(buffer)
        meta = self.meta[modelo]
        window = int(meta.get("roll_window", 6))
        raw: Dict[str, float] = {}

        for comp in COMPONENTES:
            cfg = meta["componentes"][comp]
            base = cfg["base_features"]
            feature_cols = cfg["feature_cols"]
            for col in base:
                if col not in df.columns:
                    raise ValueError(f"Coluna ausente no buffer: {col}")
            X = engenharia_features(df, base, window)
            row = X.iloc[[-1]][feature_cols]
            pipe = self.pipelines[modelo][comp]
            raw[comp] = float(-pipe.score_samples(row)[0])

        return df, raw
