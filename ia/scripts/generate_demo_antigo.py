#!/usr/bin/env python3
"""
Gera demo_antigo.csv — trator envelhecido com degradação lenta.

Narrativa (leituras a cada 5 s):
  1. Baseline elevado — desgaste natural (~60 s)
  2. Motor — atenção → alerta (~40 s)
  3. Transmissão — vibração crescente (~30 s)
  4. Hidráulico — temperatura subindo (~30 s)
  5. Freios — crítico final (~40 s)

Total: 60 leituras ≈ 5 min.
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
    HORAS_ANTIGO,
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

N_BASELINE = 12
N_MOTOR = 8
N_MOTOR_ALERTA = 8
N_TRANS = 10
N_HID = 10
N_FREIOS = 12
PASSO_SEG = 5

DEMO_INICIO = datetime(2026, 3, 20, 8, 30, 0)
EQUIPAMENTO_ID = "demo_antigo"


def _build_timestamps(n: int) -> Tuple[List[datetime], np.ndarray]:
    timestamps = [DEMO_INICIO + timedelta(seconds=i * PASSO_SEG) for i in range(n)]
    return timestamps, np.zeros(n, dtype=int)


def _gerar_base_sessao(n: int) -> pd.DataFrame:
    timestamps, day_index = _build_timestamps(n)
    reg = np.array(["campo"] * n)
    clima = _gerar_clima(day_index, timestamps, offset_temp=0.5)

    # Baseline mais alto — trator antigo
    rpm_motor = np.random.uniform(RPM_CAMPO[0] + 50, RPM_CAMPO[1] - 80, n)
    temp_arref = np.random.normal(91.0, 1.5, n)
    pres_oleo = np.random.normal(PRES_OLEO_CARGA - 0.08, 0.2, n)
    temp_ar = clima["temp_ambiente_c"] + np.random.normal(29.0, 2.0, n)
    pres_trans = np.random.normal(PRES_TRANS_NORMAL - 1.5, 0.9, n)
    rpm_tdp = np.random.normal(RPM_TDP_NOMINAL, 7.0, n)
    vib = np.random.normal(1.25, 0.18, n)
    pres_hid = np.random.normal(PRES_HID_TRABALHO - 5, 4.0, n)
    temp_hid = np.random.normal(76.0, 2.0, n)
    vazao = np.random.normal(102.0, 5.0, n)
    velocidade = np.random.uniform(VEL_CAMPO[0], VEL_CAMPO[1], n)
    angulo = np.abs(np.random.normal(5.0, 1.8, n))
    h_ini, _h_fim = HORAS_ANTIGO
    horas = np.linspace(h_ini + 80, h_ini + 80 + (n * PASSO_SEG) / 3600.0, n)

    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in timestamps],
        "equipamento_id": EQUIPAMENTO_ID,
        "regime_operacao": reg,
        "temp_arrefecimento_c": temp_arref,
        "rpm_motor": rpm_motor,
        "pressao_oleo_motor_bar": pres_oleo,
        "temp_oleo_motor_c": temp_arref + TEMP_OLEO_MOTOR_OFFSET,
        "consumo_combustivel_lh": np.random.uniform(25.0, 28.0, n),
        "temp_ar_admissao_c": temp_ar,
        "pressao_oleo_transmissao_bar": pres_trans,
        "rpm_tdp": rpm_tdp,
        "vibracao_eixo_x_mms": vib + _noise(n, 0.10),
        "vibracao_eixo_y_mms": vib + _noise(n, 0.10),
        "vibracao_eixo_z_mms": vib * 0.85 + _noise(n, 0.08),
        "pressao_hidraulica_bar": pres_hid,
        "temp_oleo_hidraulico_c": temp_hid,
        "vazao_bomba_lmin": vazao,
        "pressao_freio_bar": np.zeros(n),
        "desaceleracao_ms2": np.zeros(n),
        "temp_freio_c": np.random.normal(76.0, 2.0, n),
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


def generate_demo_antigo() -> pd.DataFrame:
    n_total = N_BASELINE + N_MOTOR + N_MOTOR_ALERTA + N_TRANS + N_HID + N_FREIOS
    df = _gerar_base_sessao(n_total)
    df = aplicar_correlacoes(df)

    s = N_BASELINE
    inject_motor_anomaly(df, s, "superaquecimento", duration=N_MOTOR, severidade="atencao")
    s += N_MOTOR
    inject_motor_anomaly(df, s, "superaquecimento", duration=N_MOTOR_ALERTA, severidade="alerta")
    s += N_MOTOR_ALERTA
    inject_transmissao_anomaly(df, s, "vibracao", duration=N_TRANS, severidade="alerta")
    s += N_TRANS
    inject_hidraulico_anomaly(df, s, "temp_alta", duration=N_HID, severidade="alerta")
    s += N_HID
    inject_freios_anomaly(df, s, "temp_alta", duration=N_FREIOS, severidade="critico")

    return df


def main() -> None:
    public_dir = os.path.join(ROOT, "..", "frontend", "public", "data")
    os.makedirs(public_dir, exist_ok=True)

    df = _finalizar(generate_demo_antigo())
    path_public = os.path.join(public_dir, "demo_antigo.csv")
    df.to_csv(path_public, index=False)

    sev = df["severidade"].value_counts().to_dict()
    print("=" * 60)
    print("demo_antigo.csv — degradação lenta")
    print(f"  Leituras: {len(df)} | Severidades: {sev}")
    print(f"  Salvo em: {path_public}")
    print("=" * 60)


if __name__ == "__main__":
    main()
