"""Testes de inferência com buffer — alinhados ao streaming da API."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml-service"))

from app.controllers.infer_controller import InferController  # noqa: E402
from app.schemas.telemetria import InferRequest, LeituraBuffer  # noqa: E402
from app.services.inference_service import (  # noqa: E402
    cor_score,
    para_score,
    score_to_level,
)
from engine import InferenceEngine  # noqa: E402

ROLL = 6


@pytest.fixture
def controller() -> InferController:
    return InferController(InferenceEngine(ROOT / "models"))


def _rows_to_buffer(df: pd.DataFrame) -> list[LeituraBuffer]:
    fields = set(LeituraBuffer.model_fields)
    return [
        LeituraBuffer(**{k: v for k, v in r.items() if k in fields})
        for r in df.to_dict(orient="records")
    ]


def _cor_esperada(score: float) -> str:
    if score <= 30:
        return "verde"
    if score <= 60:
        return "amarelo"
    if score <= 80:
        return "laranja"
    return "vermelho"


def test_para_score_interp_e_satura():
    ra = [0.0, 1.0, 5.0, 10.0]
    sa = [0.0, 15.0, 40.0, 90.0]
    assert para_score(0.0, ra, sa) == 0.0
    assert para_score(10.0, ra, sa) == 90.0
    assert para_score(100.0, ra, sa) == 90.0  # satura no topo


def test_score_to_level_coerente():
    casos = [(10, "verde"), (45, "amarelo"), (70, "laranja"), (90, "vermelho")]
    for score, cor in casos:
        nivel = score_to_level(score)
        assert nivel["cor"] == cor
        assert cor_score(score) == cor
        # verde não explica; o resto explica
        assert nivel["explicar"] == (cor != "verde")


def test_primeiros_6_normais_verde(controller: InferController):
    """Cenário real: POST com 6 leituras normais iniciais."""
    df = pd.read_csv(ROOT / "data" / "trator_novo.csv")
    normais = df[df["anomalia_injetada"] == 0].head(6)
    resp = controller.inferir(
        InferRequest(equipamento_id="trator_novo", buffer=_rows_to_buffer(normais))
    )
    assert resp.confiavel is True
    for comp in ("motor", "transmissao", "hidraulico", "freios"):
        c = resp.componentes[comp]
        assert c.score <= 30, f"{comp} score={c.score}"
        assert c.cor == "verde", f"{comp} cor={c.cor}"


def test_transmissao_anomalia_buffer_cheio(controller: InferController):
    df = pd.read_csv(ROOT / "data" / "trator_antigo.csv")
    df = df[
        (df["anomalia_injetada"] == 1) & (df["componente_anomalo"] == "transmissao")
    ].head(6)
    resp = controller.inferir(
        InferRequest(equipamento_id="trator_antigo", buffer=_rows_to_buffer(df))
    )
    t = resp.componentes["transmissao"]
    assert t.score >= 40
    assert t.cor in ("amarelo", "laranja", "vermelho")
    assert t.cor == _cor_esperada(t.score)
    assert resp.confiavel is True


def test_vibracao_crescente_manual(controller: InferController):
    base_path = ROOT / "data" / "trator_novo.csv"
    base = pd.read_csv(base_path, nrows=6)
    vibs = [(0.74, 0.84, 0.72), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (2.0, 2.0, 4.0)]
    scores: list[float] = []

    for i, (vx, vy, vz) in enumerate(vibs):
        chunk = base.iloc[: max(i + 1, 1)].copy()
        chunk.iloc[-1, chunk.columns.get_loc("vibracao_eixo_x_mms")] = vx
        chunk.iloc[-1, chunk.columns.get_loc("vibracao_eixo_y_mms")] = vy
        chunk.iloc[-1, chunk.columns.get_loc("vibracao_eixo_z_mms")] = vz
        chunk.iloc[-1, chunk.columns.get_loc("timestamp")] = f"2026-01-05T08:{10 + i * 5:02d}:00"
        resp = controller.inferir(
            InferRequest(equipamento_id="trator_novo", buffer=_rows_to_buffer(chunk))
        )
        t = resp.componentes["transmissao"]
        scores.append(t.score)
        assert t.cor == _cor_esperada(t.score)

    assert scores[-1] >= scores[0]
    assert scores[-1] >= 30
