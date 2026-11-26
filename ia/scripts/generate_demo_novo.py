#!/usr/bin/env python3
"""
Gera demo_novo.csv — trator novo em operação estável (~5 min).

Narrativa (leituras a cada 5 s no replay):
  Operação normal contínua no campo — sem degradação nem stress.

Total: 60 leituras ≈ 5 min @ 5 s/leitura.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from generate_dataset import (  # noqa: E402
    HORAS_NOVO,
    PRES_HID_TRABALHO,
    PRES_OLEO_CARGA,
    PRES_TRANS_NORMAL,
    RPM_CAMPO,
    RPM_TDP_NOMINAL,
    TEMP_OLEO_MOTOR_OFFSET,
    VEL_CAMPO,
    _finalizar,
    _gerar_clima,
    _noise,
    aplicar_correlacoes,
)

N_TOTAL = 60
PASSO_SEG = 5
DEMO_INICIO = datetime(2026, 3, 15, 10, 0, 0)
EQUIPAMENTO_ID = "demo_novo"


def _build_timestamps(n: int) -> Tuple[List[datetime], np.ndarray]:
    timestamps = [DEMO_INICIO + timedelta(seconds=i * PASSO_SEG) for i in range(n)]
    return timestamps, np.zeros(n, dtype=int)


def _gerar_base_sessao(n: int) -> pd.DataFrame:
    timestamps, day_index = _build_timestamps(n)
    reg = np.array(["campo"] * n)
    clima = _gerar_clima(day_index, timestamps, offset_temp=0.0)

    rpm_motor = np.random.uniform(RPM_CAMPO[0], RPM_CAMPO[1] - 200, n)
    temp_arref = np.random.normal(87.0, 1.0, n)
    pres_oleo = np.random.normal(PRES_OLEO_CARGA, 0.15, n)
    temp_ar = clima["temp_ambiente_c"] + np.random.normal(28.0, 1.5, n)
    pres_trans = np.random.normal(PRES_TRANS_NORMAL, 0.6, n)
    rpm_tdp = np.random.normal(RPM_TDP_NOMINAL, 5.0, n)
    vib = np.random.normal(1.0, 0.12, n)
    pres_hid = np.random.normal(PRES_HID_TRABALHO, 3.0, n)
    temp_hid = np.random.normal(70.0, 1.5, n)
    vazao = np.random.normal(110.0, 4.0, n)
    velocidade = np.random.uniform(VEL_CAMPO[0], VEL_CAMPO[1] - 0.5, n)
    angulo = np.abs(np.random.normal(3.0, 1.2, n))
    h_ini, _h_fim = HORAS_NOVO
    horas = np.linspace(h_ini + 120, h_ini + 120 + (n * PASSO_SEG) / 3600.0, n)

    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in timestamps],
        "equipamento_id": EQUIPAMENTO_ID,
        "regime_operacao": reg,
        "temp_arrefecimento_c": temp_arref,
        "rpm_motor": rpm_motor,
        "pressao_oleo_motor_bar": pres_oleo,
        "temp_oleo_motor_c": temp_arref + TEMP_OLEO_MOTOR_OFFSET,
        "consumo_combustivel_lh": np.random.uniform(23.5, 26.5, n),
        "temp_ar_admissao_c": temp_ar,
        "pressao_oleo_transmissao_bar": pres_trans,
        "rpm_tdp": rpm_tdp,
        "vibracao_eixo_x_mms": vib + _noise(n, 0.06),
        "vibracao_eixo_y_mms": vib + _noise(n, 0.06),
        "vibracao_eixo_z_mms": vib * 0.85 + _noise(n, 0.05),
        "pressao_hidraulica_bar": pres_hid,
        "temp_oleo_hidraulico_c": temp_hid,
        "vazao_bomba_lmin": vazao,
        "pressao_freio_bar": np.zeros(n),
        "desaceleracao_ms2": np.zeros(n),
        "temp_freio_c": np.random.normal(70.0, 1.5, n),
        "velocidade_kmh": velocidade,
        "angulo_inclinacao_graus": angulo,
        "horas_uso_acumuladas": horas,
        "temp_ambiente_c": clima["temp_ambiente_c"],
        "chuva_acumulada_mm": clima["chuva_acumulada_mm"],
        "umidade_solo_pct": clima["umidade_solo_pct"],
        "anomalia_injetada": 0,
        "componente_anomalo": "none",
        "severidade": "normal",
        "_day": day_index,
        "_regime": reg,
    })
    return df


def generate_demo_novo() -> pd.DataFrame:
    df = _gerar_base_sessao(N_TOTAL)
    return aplicar_correlacoes(df)


def main() -> None:
    public_dir = os.path.join(ROOT, "..", "frontend", "public", "data")
    os.makedirs(public_dir, exist_ok=True)

    df = _finalizar(generate_demo_novo())
    path_public = os.path.join(public_dir, "demo_novo.csv")
    df.to_csv(path_public, index=False)

    print("=" * 60)
    print("demo_novo.csv — operação estável")
    print(f"  Leituras: {len(df)} | equipamento_id: {EQUIPAMENTO_ID}")
    print(f"  Salvo em: {path_public}")
    print("=" * 60)


if __name__ == "__main__":
    main()
