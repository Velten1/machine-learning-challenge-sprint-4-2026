#!/usr/bin/env python3
"""
Gera demo_critico.csv — trator em terreno crítico, falha rápida (~2,5 min).

Narrativa (leituras a cada 5 s):
  1. Sensores já elevados — terreno adverso (~30 s)
  2. Atenção nos 4 componentes (~30 s)
  3. Alerta → crítico rápido (~60 s)

Total: 36 leituras ≈ 3 min @ 5 s/leitura.
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
    HORAS_CRITICO,
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
    inject_freios_anomaly,
    inject_hidraulico_anomaly,
    inject_motor_anomaly,
    inject_transmissao_anomaly,
)

OFF_TEMP = 4.0
OFF_VIB = 0.35

N_ELEVADO = 6
N_ATENCAO = 6
N_ALERTA = 8
N_CRITICO = 16
PASSO_SEG = 5

DEMO_INICIO = datetime(2026, 3, 22, 14, 0, 0)
EQUIPAMENTO_ID = "demo_critico"


def _build_timestamps(n: int) -> Tuple[List[datetime], np.ndarray]:
    timestamps = [DEMO_INICIO + timedelta(seconds=i * PASSO_SEG) for i in range(n)]
    return timestamps, np.zeros(n, dtype=int)


def _gerar_base_sessao(n: int) -> pd.DataFrame:
    timestamps, day_index = _build_timestamps(n)
    reg = np.array(["campo"] * n)
    clima = _gerar_clima(day_index, timestamps, offset_temp=OFF_TEMP)

    rpm_motor = np.random.uniform(RPM_CAMPO[0] + 100, RPM_CAMPO[1], n)
    temp_arref = np.random.normal(92.0 + OFF_TEMP, 1.8, n)
    pres_oleo = np.random.normal(PRES_OLEO_CARGA - 0.05, 0.2, n)
    temp_ar = clima["temp_ambiente_c"] + np.random.normal(30.0, 3.0, n)
    pres_trans = np.random.normal(PRES_TRANS_NORMAL - 1.0, 0.9, n)
    rpm_tdp = np.random.normal(RPM_TDP_NOMINAL, 8.0, n)
    vib = np.random.normal(1.15 + OFF_VIB, 0.20, n)
    pres_hid = np.random.normal(PRES_HID_TRABALHO - 3, 5.0, n)
    temp_hid = np.random.normal(78.0, 2.5, n)
    vazao = np.random.normal(104.0, 5.0, n)
    velocidade = np.random.uniform(VEL_CAMPO[0] + 0.5, VEL_CAMPO[1], n)
    angulo = np.abs(np.random.normal(10.0, 4.0, n))
    h_ini, _h_fim = HORAS_CRITICO
    horas = np.linspace(h_ini + 60, h_ini + 60 + (n * PASSO_SEG) / 3600.0, n)

    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in timestamps],
        "equipamento_id": EQUIPAMENTO_ID,
        "regime_operacao": reg,
        "temp_arrefecimento_c": temp_arref,
        "rpm_motor": rpm_motor,
        "pressao_oleo_motor_bar": pres_oleo,
        "temp_oleo_motor_c": temp_arref + TEMP_OLEO_MOTOR_OFFSET,
        "consumo_combustivel_lh": np.random.uniform(27.0, 29.0, n),
        "temp_ar_admissao_c": temp_ar,
        "pressao_oleo_transmissao_bar": pres_trans,
        "rpm_tdp": rpm_tdp,
        "vibracao_eixo_x_mms": vib + _noise(n, 0.12),
        "vibracao_eixo_y_mms": vib + _noise(n, 0.12),
        "vibracao_eixo_z_mms": vib * 0.85 + _noise(n, 0.10),
        "pressao_hidraulica_bar": pres_hid,
        "temp_oleo_hidraulico_c": temp_hid,
        "vazao_bomba_lmin": vazao,
        "pressao_freio_bar": np.zeros(n),
        "desaceleracao_ms2": np.zeros(n),
        "temp_freio_c": np.random.normal(78.0, 2.5, n),
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


def _injetar_todos(df: pd.DataFrame, start: int, duration: int, severidade: str) -> None:
    inject_motor_anomaly(df, start, "superaquecimento", duration=duration, severidade=severidade)
    inject_transmissao_anomaly(df, start, "vibracao", duration=duration, severidade=severidade)
    inject_hidraulico_anomaly(df, start, "temp_alta", duration=duration, severidade=severidade)
    inject_freios_anomaly(df, start, "temp_alta", duration=duration, severidade=severidade)
    end = min(start + duration, len(df))
    df.loc[start : end - 1, "angulo_inclinacao_graus"] = np.abs(
        np.random.normal(18.0, 5.0, end - start)
    )


def generate_demo_critico() -> pd.DataFrame:
    n_total = N_ELEVADO + N_ATENCAO + N_ALERTA + N_CRITICO
    df = _gerar_base_sessao(n_total)
    df = aplicar_correlacoes(df)

    s = N_ELEVADO
    _injetar_todos(df, s, N_ATENCAO, "atencao")
    s += N_ATENCAO
    _injetar_todos(df, s, N_ALERTA, "alerta")
    s += N_ALERTA
    _injetar_todos(df, s, N_CRITICO, "critico")

    return df


def main() -> None:
    public_dir = os.path.join(ROOT, "..", "frontend", "public", "data")
    os.makedirs(public_dir, exist_ok=True)

    df = _finalizar(generate_demo_critico())
    path_public = os.path.join(public_dir, "demo_critico.csv")
    df.to_csv(path_public, index=False)

    sev = df["severidade"].value_counts().to_dict()
    print("=" * 60)
    print("demo_critico.csv — crítico rápido")
    print(f"  Leituras: {len(df)} | Severidades: {sev}")
    print(f"  Salvo em: {path_public}")
    print("=" * 60)


if __name__ == "__main__":
    main()
