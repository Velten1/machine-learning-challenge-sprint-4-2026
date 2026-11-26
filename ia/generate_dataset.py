from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SEED E REPRODUTIBILIDADE
# ---------------------------------------------------------------------------
# Seed fixa garante que qualquer execução gere exatamente o mesmo CSV (demo
# reprodutível — exigência do pipeline AgroGuard, etapa 2.2).
SEED: int = 42
np.random.seed(SEED)
random.seed(SEED)


# ===========================================================================
# 1. CONSTANTES (todas as faixas de referência, com fonte do relatório)
# ===========================================================================

# --- Calendário operacional ------------------------------------------------
DATA_INICIO: datetime = datetime(2026, 1, 5, 0, 0)  # 2026-01-05 = segunda-feira
DIAS_CORRIDOS: int = 30
PASSO_MIN: int = 5                  # 1 leitura a cada 5 minutos
HORA_INICIO_OP: int = 7             # operação começa 07h00
HORA_FIM_OP: int = 18               # operação termina 18h00
ALMOCO_INICIO: int = 12             # pausa almoço (ralenti) 12h00–13h00
ALMOCO_FIM: int = 13
DIR_SAIDA: str = "data"

# --- MOTOR -----------------------------------------------------------------
# Temperatura do líquido de arrefecimento.
# [OFICIAL] Manual PowerTech 4045/6068: faixa normal 82–95 °C; reduzir carga
# acima de 112 °C; termostato abre a 85 °C e fica 100% aberto a 97 °C.
TEMP_ARREF_NORMAL: float = 90.0
TEMP_ARREF_ATENCAO: float = 100.0   # acima do normal de operação
TEMP_ARREF_ALERTA: float = 105.0
TEMP_ARREF_CRITICO: float = 110.0   # próximo do limite de 112 °C (parar motor)

# Rotação do motor por regime.
# [OFICIAL] Nominal 2100 rpm; campo na faixa de potência constante 1600–2100;
# transporte EcoShift 1580–1680 rpm (Brochura 6M MY21); ralenti de fábrica
# ~800–850 rpm (Manual PowerTech) — correlação 8 do pipeline pede 875–900.
RPM_CAMPO: Tuple[float, float] = (1600.0, 2100.0)
RPM_TRANSPORTE: Tuple[float, float] = (1580.0, 1680.0)
RPM_RALENTI: Tuple[float, float] = (875.0, 900.0)

# Pressão do óleo do motor.
# [OFICIAL] Manual PowerTech 4045: plena carga/rotação nominal 3,45 ± 1,03 bar
# (50 ± 15 psi); mínima em marcha lenta 1,03 bar (15 psi).
PRES_OLEO_CARGA: float = 3.45
PRES_OLEO_IDLE_MIN: float = 1.03

# Consumo de combustível — curva Nebraska Test (6150M, correlação 2 do pipeline).
# [ACADÊMICO] Nebraska/OECD: carga máxima ~29,6 L/h | 25% ~18,5 L/h | ralenti ~11,0 L/h.
FUEL_LOAD_PTS: Tuple[float, float, float] = (0.0, 0.25, 1.0)
FUEL_LH_PTS: Tuple[float, float, float] = (11.0, 18.5, 29.6)

# Temperatura do óleo do motor.
# [OFICIAL] Pressão de óleo é especificada com o óleo a 115 °C em plena carga
# (Manual PowerTech) → óleo opera ~12–18 °C acima do arrefecimento.
TEMP_OLEO_MOTOR_OFFSET: float = 14.0

# --- TRANSMISSÃO -----------------------------------------------------------
# Pressão do óleo da transmissão.
# [TÉCNICO] DTC de baixa pressão dispara abaixo de ~13,8 bar (200 psi)
# (GreenTractorTalk / justanswer). Normal de operação ~16–20 bar.
PRES_TRANS_NORMAL: float = 18.0
PRES_TRANS_ALERTA: float = 13.8     # < 200 psi
PRES_TRANS_CRITICO: float = 12.0

# RPM de saída da TDP.
# [OFICIAL] TDP 540/1000; engine ~1948–1977 rpm na rotação nominal da TDP.
# Modelado como rotação do eixo de saída (540 nominal quando engatada em campo).
RPM_TDP_NOMINAL: float = 540.0

# Vibração nos eixos (mm/s RMS).
# Estimativa normativa: ISO 20816-3:2022 Grupo 2 — JD não publica vibração de
# eixo para a 6M. Zona A 0,0–1,4 | B 1,4–2,8 | C 2,8–4,5 | D >4,5.
VIB_ZONA_A: float = 1.4
VIB_ZONA_B: float = 2.8
VIB_ZONA_C: float = 4.5

# --- HIDRÁULICO ------------------------------------------------------------
# Pressão hidráulica.
# [OFICIAL] PFC closed-center: trabalho até 200 bar (2900 psi); standby da bomba
# ~218 bar (limite swash plate, OMHXE29688); standby de transporte ~41 bar.
PRES_HID_TRABALHO: float = 185.0
PRES_HID_STANDBY: float = 41.0
PRES_HID_ATENCAO: float = 155.0
PRES_HID_ALERTA: float = 150.0      # < 150 bar sob carga = anomalia (pipeline)
PRES_HID_CRITICO: float = 140.0

# Temperatura do óleo hidráulico.
# [TÉCNICO] Referência hidráulica/Hy-Gard: ideal 38–60 °C, alerta 60–82 °C,
# crítico >82 °C; DTC de óleo quente (SPN 1638.16) usado a >93 °C no pipeline.
TEMP_HID_NORMAL: float = 70.0
TEMP_HID_ATENCAO: float = 88.0
TEMP_HID_ALERTA: float = 93.0
TEMP_HID_CRITICO: float = 100.0

# Vazão da bomba.
# [OFICIAL] PFC base 114 L/min (30 gpm); PC 80 L/min; standby = baixa vazão.
VAZAO_PFC_BASE: float = 114.0

# --- FREIOS ----------------------------------------------------------------
# Pressão de acionamento.
# Estimativa normativa — sem dado oficial JD para 6M (sistemas de disco molhado
# hidráulico agrícola operam tipicamente em 20–40 bar).
PRES_FREIO_MIN: float = 20.0
PRES_FREIO_MAX: float = 40.0

# Temperatura dos freios.
# Derivado do Hy-Gard — discos banhados no mesmo óleo da transmissão.
# Normal 60–80 °C, alarme >93 °C.
TEMP_FREIO_NORMAL_MIN: float = 60.0
TEMP_FREIO_NORMAL_MAX: float = 80.0
TEMP_FREIO_ALARME: float = 93.0

# Desaceleração.
# Regulamento EU 167/2013 — frenagem mínima agrícola. Normal 2,5–4,0 m/s²,
# referência 3,5 m/s².
DESACEL_MIN: float = 2.5
DESACEL_MAX: float = 4.0

# --- OPERAÇÃO GERAL --------------------------------------------------------
# Velocidade de campo — não é spec do fabricante; definida por regime (pipeline).
VEL_CAMPO: Tuple[float, float] = (5.0, 8.0)        # preparo 6–8 / plantio 5–7
VEL_TRANSPORTE: Tuple[float, float] = (25.0, 40.0)  # transporte 25–40 km/h

# Ângulo de inclinação lateral.
# [ACADÊMICO] Regra prática ~15° (TractorByNet); limite teórico de tombamento
# ~31° (estudos de rollover). Operação normal mantém ângulos baixos.
ANG_NORMAL_MEDIA: float = 3.0
ANG_CRITICO_MAX: float = 25.0

# Horas de uso acumuladas por perfil (início → fim em 30 dias).
HORAS_NOVO: Tuple[float, float] = (250.0, 700.0)
HORAS_ANTIGO: Tuple[float, float] = (4200.0, 4650.0)
HORAS_CRITICO: Tuple[float, float] = (1800.0, 2250.0)

# --- Severidade ------------------------------------------------------------
SEV_RANK: Dict[str, int] = {"normal": 0, "atencao": 1, "alerta": 2, "critico": 3}

# Severidades injetáveis (ordem crescente) para a "escada de severidade".
SEV_LADDER: Tuple[str, str, str] = ("atencao", "alerta", "critico")

# Bandas de magnitude por severidade, por tipo de anomalia. Cada banda é
# fisicamente fundamentada (mesmas fontes das constantes acima) e foi separada
# de forma que atencao < alerta < critico produzam scores brutos do Isolation
# Forest claramente distinguíveis — requisito da calibração por severidade
# (PASSO 2). A separação entre bandas equivale a ~1 faixa física (≈1σ+) por
# degrau, atendendo ao alvo atencao ~1,5–2σ / alerta ~2,5–3σ / critico ~4σ+.
#
# As bandas são separadas por ~1σ+ da linha de base entre degraus para que o
# Isolation Forest (e o z-score do texto) consigam RANQUEAR a gravidade — bandas
# coladas (degraus de ~5 °C) saturavam o IF e critico ≈ alerta no score.
#
# MOTOR — temp_arrefecimento_c. Normal de operação chega a ~100 °C (soma das
# correlações rpm+ambiente), então atenção começa acima disso. [OFICIAL] Manual
# PowerTech: parada recomendada >112 °C; crítico = falha de arrefecimento com
# fervura (>119 °C), acima do limite de parada.
BANDA_MOTOR_TEMP: Dict[str, Tuple[float, float]] = {
    "atencao": (103.0, 107.0),
    "alerta": (110.0, 115.0),
    "critico": (119.0, 126.0),
}
# TRANSMISSÃO — vibração mm/s [ISO 20816-3:2022] Zona B (atenção),
# Zona C (alerta), Zona D severa (crítico).
BANDA_TRANS_VIB: Dict[str, Tuple[float, float]] = {
    "atencao": (1.8, 2.6),    # Zona B
    "alerta": (3.2, 4.3),     # Zona C
    "critico": (5.5, 8.5),    # Zona D severa
}
# HIDRÁULICO — temp_oleo_hidraulico_c [TÉCNICO/Hy-Gard] alarme >93 °C; crítico =
# superaquecimento severo do óleo (>112 °C).
BANDA_HID_TEMP: Dict[str, Tuple[float, float]] = {
    "atencao": (90.0, 95.0),
    "alerta": (100.0, 106.0),
    "critico": (112.0, 120.0),
}
# FREIOS — temp_freio_c [Hy-Gard derivado] alarme >93 °C; crítico = fade térmico
# severo dos discos (>113 °C).
BANDA_FREIO_TEMP: Dict[str, Tuple[float, float]] = {
    "atencao": (88.0, 94.0),
    "alerta": (100.0, 107.0),
    "critico": (113.0, 122.0),
}

# Ordem final das colunas do CSV (conforme especificação do pipeline).
COLUNAS: List[str] = [
    "timestamp", "equipamento_id", "regime_operacao",
    "temp_arrefecimento_c", "rpm_motor", "pressao_oleo_motor_bar",
    "temp_oleo_motor_c", "consumo_combustivel_lh", "temp_ar_admissao_c",
    "pressao_oleo_transmissao_bar", "rpm_tdp",
    "vibracao_eixo_x_mms", "vibracao_eixo_y_mms", "vibracao_eixo_z_mms",
    "pressao_hidraulica_bar", "temp_oleo_hidraulico_c", "vazao_bomba_lmin",
    "pressao_freio_bar", "desaceleracao_ms2", "temp_freio_c",
    "velocidade_kmh", "angulo_inclinacao_graus", "horas_uso_acumuladas",
    "temp_ambiente_c", "chuva_acumulada_mm", "umidade_solo_pct",
    "anomalia_injetada", "componente_anomalo", "severidade",
]

# Faixas esperadas (texto) para o relatório de validação final.
FAIXAS_ESPERADAS: Dict[str, str] = {
    "temp_arrefecimento_c": "normal 82-95 C; >112 parar [OFICIAL]",
    "rpm_motor": "ralenti 875-900; campo 1600-2100; transporte 1580-1680 [OFICIAL]",
    "pressao_oleo_motor_bar": "idle >=1.03; carga 3.45+-1.03 [OFICIAL]",
    "temp_oleo_motor_c": "~oleo a 115 C plena carga [OFICIAL]",
    "consumo_combustivel_lh": "ralenti ~11; max ~29.6 (6150M Nebraska) [ACADEMICO]",
    "temp_ar_admissao_c": "ambiente + boost [J1939 estimativa]",
    "pressao_oleo_transmissao_bar": "normal ~16-20; alarme <13.8 [TECNICO]",
    "rpm_tdp": "540 nominal em campo; 0 fora de campo [OFICIAL]",
    "vibracao_eixo_x_mms": "A 0-1.4 / B 1.4-2.8 / C 2.8-4.5 / D >4.5 [ISO 20816-3]",
    "vibracao_eixo_y_mms": "A 0-1.4 / B 1.4-2.8 / C 2.8-4.5 / D >4.5 [ISO 20816-3]",
    "vibracao_eixo_z_mms": "A 0-1.4 / B 1.4-2.8 / C 2.8-4.5 / D >4.5 [ISO 20816-3]",
    "pressao_hidraulica_bar": "trabalho ~185; max 200; standby ~41 [OFICIAL]",
    "temp_oleo_hidraulico_c": "normal 60-82; alarme >93 [TECNICO/Hy-Gard]",
    "vazao_bomba_lmin": "PFC 114 base; standby baixo [OFICIAL]",
    "pressao_freio_bar": "20-40 (estimativa normativa)",
    "desaceleracao_ms2": "2.5-4.0; ref 3.5 [EU 167/2013]",
    "temp_freio_c": "normal 60-80; >93 alarme [Hy-Gard]",
    "velocidade_kmh": "campo 5-8; transporte 25-40 (normativa)",
    "angulo_inclinacao_graus": "pratico <=15; tombamento ~31 [ACADEMICO]",
    "horas_uso_acumuladas": "novo 250-700 / antigo 4200-4650 / critico 1800-2250",
    "temp_ambiente_c": "verao BR ~20-38 (mock)",
    "chuva_acumulada_mm": "0-40 (mock)",
    "umidade_solo_pct": "20-95 (mock)",
}

# Limites físicos plausíveis (clip final — zero NaN/inf, nada negativo).
LIMITES_FISICOS: Dict[str, Tuple[float, float]] = {
    "temp_arrefecimento_c": (0.0, 130.0),
    "rpm_motor": (0.0, 2500.0),
    "pressao_oleo_motor_bar": (0.0, 6.0),
    "temp_oleo_motor_c": (0.0, 150.0),
    "consumo_combustivel_lh": (0.0, 45.0),
    "temp_ar_admissao_c": (0.0, 90.0),
    "pressao_oleo_transmissao_bar": (0.0, 30.0),
    "rpm_tdp": (0.0, 1200.0),
    "vibracao_eixo_x_mms": (0.0, 12.0),
    "vibracao_eixo_y_mms": (0.0, 12.0),
    "vibracao_eixo_z_mms": (0.0, 12.0),
    "pressao_hidraulica_bar": (0.0, 220.0),
    "temp_oleo_hidraulico_c": (0.0, 130.0),
    "vazao_bomba_lmin": (0.0, 160.0),
    "pressao_freio_bar": (0.0, 60.0),
    "desaceleracao_ms2": (0.0, 8.0),
    "temp_freio_c": (0.0, 130.0),
    "velocidade_kmh": (0.0, 50.0),
    "angulo_inclinacao_graus": (0.0, 35.0),
    "horas_uso_acumuladas": (0.0, 60000.0),
    "temp_ambiente_c": (0.0, 50.0),
    "chuva_acumulada_mm": (0.0, 120.0),
    "umidade_solo_pct": (0.0, 100.0),
}


# ===========================================================================
# UTILITÁRIOS
# ===========================================================================

def _noise(n: int, sd: float) -> np.ndarray:
    """Ruído gaussiano de média zero e desvio `sd` para `n` amostras."""
    return np.random.normal(0.0, sd, n)


def _classificar_severidade(*severidades: str) -> str:
    """Retorna a pior severidade dentre as fornecidas (ordem SEV_RANK)."""
    return max(severidades, key=lambda s: SEV_RANK[s])


def _set_label(df: pd.DataFrame, idx: int, componente: str, severidade: str) -> None:
    """Aplica um rótulo de anomalia a uma linha, mantendo sempre a pior severidade.

    Args:
        df: DataFrame alvo (modificado in place).
        idx: índice posicional da linha.
        componente: componente anômalo (motor|transmissao|hidraulico|freios).
        severidade: severidade do evento (atencao|alerta|critico).
    """
    atual = df.at[idx, "severidade"]
    if SEV_RANK[severidade] > SEV_RANK[atual]:
        df.at[idx, "severidade"] = severidade
        df.at[idx, "componente_anomalo"] = componente
        df.at[idx, "anomalia_injetada"] = 1


# ===========================================================================
# CALENDÁRIO OPERACIONAL
# ===========================================================================

def build_calendar() -> Tuple[List[datetime], List[str], np.ndarray]:
    """Constrói a linha do tempo operacional dos 30 dias.

    Regras (pipeline AgroGuard):
        - Segunda a sábado, das 07h00 às 18h00, 1 leitura a cada 5 min.
        - Pausa de almoço 12h00–13h00 em regime ralenti.
        - Domingo: equipamento parado (sem leituras).
        - Fora do horário operacional: sem leituras.

    Returns:
        timestamps: lista de datetime por leitura.
        regimes: lista de regime ('campo'|'transporte'|'ralenti') por leitura.
        day_index: array com o índice do dia (0..29) por leitura.
    """
    timestamps: List[datetime] = []
    regimes: List[str] = []
    day_index: List[int] = []

    for d in range(DIAS_CORRIDOS):
        dia = DATA_INICIO + timedelta(days=d)
        if dia.weekday() == 6:  # domingo (Monday=0 ... Sunday=6) → sem leituras
            continue
        t = dia.replace(hour=HORA_INICIO_OP, minute=0)
        fim = dia.replace(hour=HORA_FIM_OP, minute=0)
        while t < fim:
            h = t.hour
            if ALMOCO_INICIO <= h < ALMOCO_FIM:
                regime = "ralenti"               # pausa de almoço
            elif HORA_INICIO_OP <= h < 10:
                regime = "campo"                 # 07h–10h: preparo/plantio
            elif 10 <= h < ALMOCO_INICIO:
                regime = "transporte"            # 10h–12h: transporte entre talhões
            elif ALMOCO_FIM <= h < 16:
                regime = "campo"                 # 13h–16h: campo
            else:
                regime = "transporte"            # 16h–18h: transporte de retorno
            timestamps.append(t)
            regimes.append(regime)
            day_index.append(d)
            t += timedelta(minutes=PASSO_MIN)

    return timestamps, regimes, np.asarray(day_index, dtype=int)


# ===========================================================================
# CLIMA MOCKADO (APIs externas simuladas dentro do gerador)
# ===========================================================================

def _gerar_clima(day_index: np.ndarray, timestamps: List[datetime],
                 offset_temp: float) -> Dict[str, np.ndarray]:
    """Gera variáveis ambientais mockadas (INMET/ANA/EMBRAPA simulados).

    Sem integração real — mock interno conforme decisão de arquitetura do
    AgroGuard. Modela ciclo diurno de temperatura, eventos de chuva diários e
    umidade do solo com atraso de 1–2 dias após a chuva (correlação 4).

    Args:
        day_index: índice do dia por leitura.
        timestamps: datetime por leitura.
        offset_temp: deslocamento de temperatura do perfil (crítico = mais quente).

    Returns:
        dict com arrays temp_ambiente_c, chuva_acumulada_mm, umidade_solo_pct.
    """
    n = len(timestamps)

    # Nível diário (verão BR, jan/fev ~ 28 °C base).
    temp_base_dia = np.random.normal(28.0 + offset_temp, 2.0, DIAS_CORRIDOS)
    # Chuva diária acumulada (24h): ~65% dos dias sem chuva.
    chuva_dia = np.where(
        np.random.random(DIAS_CORRIDOS) < 0.65,
        0.0,
        np.random.uniform(3.0, 40.0, DIAS_CORRIDOS),
    )
    # Umidade do solo responde à chuva de d-1 e d-2 (atraso de 1–2 dias).
    umidade_dia = np.zeros(DIAS_CORRIDOS)
    for d in range(DIAS_CORRIDOS):
        chuva_d1 = chuva_dia[d - 1] if d - 1 >= 0 else 0.0
        chuva_d2 = chuva_dia[d - 2] if d - 2 >= 0 else 0.0
        umidade_dia[d] = 35.0 + 0.6 * chuva_d1 + 0.4 * chuva_d2  # correlação 4

    temp_ambiente = np.empty(n)
    chuva_acumulada = np.empty(n)
    umidade_solo = np.empty(n)
    for i, (d, ts) in enumerate(zip(day_index, timestamps)):
        # Ciclo diurno: pico no início da tarde, mínimo de manhã cedo.
        diurno = 5.0 * np.sin((ts.hour - 6) / 12.0 * np.pi)
        temp_ambiente[i] = temp_base_dia[d] + diurno
        chuva_acumulada[i] = chuva_dia[d]
        umidade_solo[i] = umidade_dia[d]

    temp_ambiente += _noise(n, 0.8)
    chuva_acumulada = np.clip(chuva_acumulada + _noise(n, 0.4), 0.0, None)
    umidade_solo = np.clip(umidade_solo + _noise(n, 1.0), 20.0, 95.0)

    return {
        "temp_ambiente_c": temp_ambiente,
        "chuva_acumulada_mm": chuva_acumulada,
        "umidade_solo_pct": umidade_solo,
    }


# ===========================================================================
# 2. GERAÇÃO BASE (estado saudável por regime, antes das anomalias)
# ===========================================================================

def _gerar_base(profile: str) -> pd.DataFrame:
    """Gera o DataFrame base (operação saudável) de um perfil.

    Produz todas as colunas em estado NORMAL, com ruído gaussiano realista e
    valores ditados pelo regime operacional. Correlações e anomalias são
    aplicadas depois. Offsets de perfil deixam o trator crítico mais quente,
    mais vibrante e em terreno mais inclinado já na linha de base.

    Args:
        profile: 'novo' | 'antigo' | 'critico'.

    Returns:
        DataFrame base com rótulos zerados (anomalia_injetada=0).
    """
    timestamps, regimes, day_index = build_calendar()
    n = len(timestamps)
    reg = np.asarray(regimes)
    campo = reg == "campo"
    transp = reg == "transporte"
    ral = reg == "ralenti"

    # Offsets por perfil (crítico = condições hostis na linha de base).
    off_temp = 4.0 if profile == "critico" else 0.0
    off_vib = 0.3 if profile == "critico" else 0.0

    # --- MOTOR -------------------------------------------------------------
    rpm_motor = np.empty(n)
    rpm_motor[campo] = np.random.uniform(*RPM_CAMPO, campo.sum())
    rpm_motor[transp] = np.random.uniform(*RPM_TRANSPORTE, transp.sum())
    rpm_motor[ral] = np.random.uniform(*RPM_RALENTI, ral.sum())  # correlação 8

    # Temperatura base por regime (mantida abaixo de 100 °C — operação normal).
    temp_arref = np.empty(n)
    temp_arref[campo] = np.random.normal(88.0 + off_temp, 1.5, campo.sum())
    temp_arref[transp] = np.random.normal(86.0 + off_temp, 1.5, transp.sum())
    temp_arref[ral] = np.random.normal(83.0 + off_temp, 1.5, ral.sum())

    # Pressão do óleo do motor: carga ~3,45 bar; ralenti ~1,05 bar (>=1,03).
    pres_oleo = np.empty(n)
    pres_oleo[~ral] = np.random.normal(PRES_OLEO_CARGA, 0.20, (~ral).sum())
    pres_oleo[ral] = np.random.normal(1.05, 0.03, ral.sum())  # correlação 8

    # Temperatura do ar de admissão: ambiente + aquecimento por boost.
    # (preenchida após o clima; placeholder aqui)
    temp_ar = np.zeros(n)

    # --- TRANSMISSÃO -------------------------------------------------------
    pres_trans = np.empty(n)
    pres_trans[~ral] = np.random.normal(PRES_TRANS_NORMAL, 1.0, (~ral).sum())
    pres_trans[ral] = np.random.normal(15.0, 0.8, ral.sum())

    # RPM da TDP: ~540 quando engatada em campo; 0 em transporte/ralenti.
    rpm_tdp = np.zeros(n)
    rpm_tdp[campo] = np.random.normal(RPM_TDP_NOMINAL, 8.0, campo.sum())

    # Vibração nos eixos (mm/s RMS) — perfil novo permanece na Zona A.
    def _vib(media: float) -> np.ndarray:
        v = np.empty(n)
        v[campo] = np.random.normal(media + off_vib, 0.20, campo.sum())
        v[transp] = np.random.normal(media - 0.2 + off_vib, 0.18, transp.sum())
        v[ral] = np.random.normal(0.5 + off_vib, 0.12, ral.sum())
        return np.clip(v, 0.0, None)

    vib_x = _vib(1.10)
    vib_y = _vib(1.05)
    vib_z = _vib(0.85)

    # --- HIDRÁULICO --------------------------------------------------------
    pres_hid = np.empty(n)
    pres_hid[campo] = np.random.normal(PRES_HID_TRABALHO, 5.0, campo.sum())
    pres_hid[transp] = np.random.normal(PRES_HID_STANDBY, 3.0, transp.sum())  # corr.7
    pres_hid[ral] = np.random.normal(38.0, 3.0, ral.sum())

    vazao = np.empty(n)
    vazao[campo] = np.random.normal(105.0, 6.0, campo.sum())   # até 114 L/min
    vazao[transp] = np.random.normal(25.0, 8.0, transp.sum())  # standby
    vazao[ral] = np.random.normal(12.0, 4.0, ral.sum())
    vazao = np.clip(vazao, 0.0, None)

    # Temperatura do óleo hidráulico (preenchida após o clima — depende do ambiente).
    temp_hid = np.zeros(n)

    # --- FREIOS ------------------------------------------------------------
    # Freadas NORMAIS ocorrem de forma intermitente (transporte > campo). Banda
    # suave (18–28 bar / 2,0–3,2 m/s²) para que uma frenagem rotineira não pareça
    # falha ao Isolation Forest — antes picos de até 40 bar em leituras normais
    # isoladas faziam os freios irem a amarelo sem anomalia (visível no crítico).
    # Frenagens fortes/irregulares ficam reservadas às anomalias injetadas.
    pres_freio = np.zeros(n)
    desacel = np.zeros(n)
    prob_freada = np.where(transp, 0.12, np.where(campo, 0.04, 0.0))
    freando = np.random.random(n) < prob_freada
    pres_freio[freando] = np.random.uniform(18.0, 28.0, freando.sum())
    desacel[freando] = np.random.uniform(2.0, 3.2, freando.sum())

    temp_freio = np.zeros(n)  # definido por correlação 3 (compartilha óleo Hy-Gard)

    # --- OPERAÇÃO GERAL ----------------------------------------------------
    velocidade = np.zeros(n)
    velocidade[campo] = np.random.uniform(*VEL_CAMPO, campo.sum())
    velocidade[transp] = np.random.uniform(*VEL_TRANSPORTE, transp.sum())

    if profile == "critico":
        # Terreno irregular: inclinação frequentemente elevada.
        angulo = np.abs(np.random.normal(8.0, 5.0, n))
    else:
        angulo = np.abs(np.random.normal(ANG_NORMAL_MEDIA, 2.0, n))

    # Horas acumuladas: interpolação linear monotônica início→fim do perfil.
    h_ini, h_fim = {
        "novo": HORAS_NOVO, "antigo": HORAS_ANTIGO, "critico": HORAS_CRITICO,
    }[profile]
    horas = np.linspace(h_ini, h_fim, n)

    # --- CLIMA -------------------------------------------------------------
    clima = _gerar_clima(day_index, timestamps, off_temp)

    # Temperatura do ar de admissão = ambiente + aquecimento por regime.
    temp_ar[campo] = clima["temp_ambiente_c"][campo] + np.random.normal(28.0, 4.0, campo.sum())
    temp_ar[transp] = clima["temp_ambiente_c"][transp] + np.random.normal(18.0, 3.0, transp.sum())
    temp_ar[ral] = clima["temp_ambiente_c"][ral] + np.random.normal(6.0, 2.0, ral.sum())

    # Temperatura do óleo hidráulico: base + carga + acoplamento com o ambiente.
    temp_hid = np.full(n, TEMP_HID_NORMAL) + _noise(n, 3.0)
    temp_hid[campo] += 5.0
    temp_hid += 0.2 * (clima["temp_ambiente_c"] - 28.0)

    # Óleo do motor mais quente que o arrefecimento.
    temp_oleo_motor = temp_arref + TEMP_OLEO_MOTOR_OFFSET + _noise(n, 1.5)

    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in timestamps],
        "equipamento_id": f"trator_{profile}",
        "regime_operacao": reg,
        "temp_arrefecimento_c": temp_arref,
        "rpm_motor": rpm_motor,
        "pressao_oleo_motor_bar": pres_oleo,
        "temp_oleo_motor_c": temp_oleo_motor,
        "consumo_combustivel_lh": np.zeros(n),   # definido na correlação 2
        "temp_ar_admissao_c": temp_ar,
        "pressao_oleo_transmissao_bar": pres_trans,
        "rpm_tdp": rpm_tdp,
        "vibracao_eixo_x_mms": vib_x,
        "vibracao_eixo_y_mms": vib_y,
        "vibracao_eixo_z_mms": vib_z,
        "pressao_hidraulica_bar": pres_hid,
        "temp_oleo_hidraulico_c": temp_hid,
        "vazao_bomba_lmin": vazao,
        "pressao_freio_bar": pres_freio,
        "desaceleracao_ms2": desacel,
        "temp_freio_c": temp_freio,
        "velocidade_kmh": velocidade,
        "angulo_inclinacao_graus": angulo,
        "horas_uso_acumuladas": horas,
        "temp_ambiente_c": clima["temp_ambiente_c"],
        "chuva_acumulada_mm": clima["chuva_acumulada_mm"],
        "umidade_solo_pct": clima["umidade_solo_pct"],
        "anomalia_injetada": 0,
        "componente_anomalo": "none",
        "severidade": "normal",
        "_day": day_index,             # auxiliar (removido antes de salvar)
        "_regime": reg,                # auxiliar para injeção de anomalias
    })
    return df


# ===========================================================================
# 3. FUNÇÕES DE CORRELAÇÃO (aplicadas após a base, antes das anomalias)
# ===========================================================================

def corr_rpm_temperatura(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação 1: RPM alto eleva a temperatura de arrefecimento com atraso.

    Atraso de 2 leituras (~10 min). Fonte: comportamento físico documentado
    dos motores PowerTech 4045/6068 (carga térmica acompanha a rotação).
    """
    rpm_atrasado = df["rpm_motor"].shift(2).fillna(df["rpm_motor"].iloc[0])
    df["temp_arrefecimento_c"] = df["temp_arrefecimento_c"] + 0.008 * (rpm_atrasado - 1500.0)
    df["temp_oleo_motor_c"] = df["temp_arrefecimento_c"] + TEMP_OLEO_MOTOR_OFFSET
    return df


def corr_carga_consumo(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação 2: carga alta eleva o consumo seguindo a curva Nebraska (6150M).

    carga máxima → ~29,6 L/h | 25% → ~18,5 L/h | ralenti → ~11,0 L/h.
    A fração de carga é derivada do regime (ralenti=0, transporte parcial,
    campo alta). Fonte: Nebraska Tractor Test 2080 (John Deere 6150M).
    """
    reg = df["_regime"].to_numpy()
    n = len(df)
    carga = np.zeros(n)
    carga[reg == "transporte"] = np.random.uniform(0.35, 0.55, (reg == "transporte").sum())
    carga[reg == "campo"] = np.random.uniform(0.70, 1.00, (reg == "campo").sum())
    consumo = np.interp(carga, FUEL_LOAD_PTS, FUEL_LH_PTS) + _noise(n, 0.4)
    df["consumo_combustivel_lh"] = np.clip(consumo, 0.0, None)
    return df


def corr_hidraulico_freio_temp(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação 3: temperatura do óleo hidráulico e dos freios andam juntas.

    Transmissão e freios compartilham o mesmo reservatório de óleo Hy-Gard,
    então a temperatura dos freios segue a do óleo hidráulico (com pequeno
    offset e ruído).
    """
    df["temp_freio_c"] = df["temp_oleo_hidraulico_c"] - 3.0 + _noise(len(df), 1.5)
    return df


def corr_chuva_umidade_solo(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação 4: chuva alta eleva a umidade do solo com atraso de 1–2 dias.

    Já modelada na geração do clima (umidade do dia depende da chuva de d-1/d-2).
    Esta função existe para deixar a correlação explícita no pipeline; nenhuma
    transformação adicional é necessária aqui.
    """
    return df


def corr_ambiente_temperatura(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação 5: temperatura ambiente alta eleva a linha de base do arrefecimento."""
    df["temp_arrefecimento_c"] = df["temp_arrefecimento_c"] + 0.25 * (df["temp_ambiente_c"] - 28.0)
    df["temp_oleo_motor_c"] = df["temp_arrefecimento_c"] + TEMP_OLEO_MOTOR_OFFSET
    return df


def corr_vibracao_tdp(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação 6: vibração alta torna o RPM da TDP instável (desgaste mecânico).

    Aplica ruído ao rpm_tdp proporcional ao excesso de vibração, apenas onde a
    TDP está engatada (rpm_tdp > 0).
    """
    vib_max = df[["vibracao_eixo_x_mms", "vibracao_eixo_y_mms", "vibracao_eixo_z_mms"]].max(axis=1)
    engatada = df["rpm_tdp"] > 0
    excesso = np.clip(vib_max - 1.5, 0.0, None)
    ruido = np.random.normal(0.0, 1.0, len(df)) * excesso * 15.0
    df.loc[engatada, "rpm_tdp"] = df.loc[engatada, "rpm_tdp"] + ruido[engatada.to_numpy()]
    df["rpm_tdp"] = np.clip(df["rpm_tdp"], 0.0, None)
    return df


def aplicar_correlacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as correlações obrigatórias na ordem do pipeline."""
    df = corr_rpm_temperatura(df)        # 1
    df = corr_carga_consumo(df)          # 2
    df = corr_hidraulico_freio_temp(df)  # 3
    df = corr_chuva_umidade_solo(df)     # 4
    df = corr_ambiente_temperatura(df)   # 5
    df = corr_vibracao_tdp(df)           # 6
    # Correlações 7 (transporte) e 8 (ralenti) já são impostas na geração base.
    return df


# ===========================================================================
# 4. FUNÇÕES DE INJEÇÃO DE ANOMALIA POR COMPONENTE
# ===========================================================================

def inject_motor_anomaly(df: pd.DataFrame, start_idx: int, tipo: str,
                         duration: int = 24, peak: float = 112.0,
                         start_temp: float = 94.0,
                         severidade: Optional[str] = None,
                         events: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Injeta anomalia de MOTOR a partir de `start_idx`.

    Tipos:
        'superaquecimento' — temp_arrefecimento sobe de `start_temp` até `peak`
            em `duration` leituras. Severidade: atencao >100, alerta >105,
            critico >110. Fonte: Manual JD OMN401512; ECU 110.00.
        'pressao_oleo' — pressao_oleo_motor cai abaixo de 1,03 bar (mín. idle)
            por `duration` leituras. Fonte: Manual PowerTech 4045/6068.
        'rpm_instavel' — oscilação de ±200 rpm fora do padrão. Severidade atencao.

    Args:
        df: DataFrame alvo (in place).
        start_idx: índice posicional de início.
        tipo: tipo de anomalia.
        duration: duração em número de leituras.
        peak: temperatura de pico (apenas 'superaquecimento').
        start_temp: temperatura inicial da rampa (apenas 'superaquecimento').
        severidade: se informado (atencao|alerta|critico) e tipo=='superaquecimento',
            usa banda PLANA de temperatura da severidade (BANDA_MOTOR_TEMP) e força
            esse rótulo — usado pela escada de severidade da calibração (PASSO 2).
        events: lista para registrar o evento (start/end/componente/var/sev).
    """
    end = min(start_idx + duration, len(df))
    ts_ini, ts_fim = df.at[start_idx, "timestamp"], df.at[end - 1, "timestamp"]

    if tipo == "superaquecimento" and severidade in BANDA_MOTOR_TEMP:
        lo, hi = BANDA_MOTOR_TEMP[severidade]
        for i in range(start_idx, end):
            t = np.random.uniform(lo, hi)
            df.at[i, "temp_arrefecimento_c"] = t
            df.at[i, "temp_oleo_motor_c"] = t + TEMP_OLEO_MOTOR_OFFSET
            _set_label(df, i, "motor", severidade)
        var = "temp_arrefecimento_c"

    elif tipo == "superaquecimento":
        for k, i in enumerate(range(start_idx, end)):
            frac = k / max(1, (end - start_idx - 1))
            t = start_temp + (peak - start_temp) * frac + np.random.normal(0.0, 0.5)
            df.at[i, "temp_arrefecimento_c"] = t
            df.at[i, "temp_oleo_motor_c"] = t + TEMP_OLEO_MOTOR_OFFSET
            sev = ("critico" if t > TEMP_ARREF_CRITICO else
                   "alerta" if t > TEMP_ARREF_ALERTA else
                   "atencao" if t > TEMP_ARREF_ATENCAO else "normal")
            if sev != "normal":
                _set_label(df, i, "motor", sev)
        var = "temp_arrefecimento_c"

    elif tipo == "pressao_oleo":
        for i in range(start_idx, end):
            p = np.random.uniform(0.7, 0.95)  # abaixo do mínimo de idle (1,03 bar)
            df.at[i, "pressao_oleo_motor_bar"] = p
            sev = "critico" if p < 0.8 else "alerta"
            _set_label(df, i, "motor", sev)
        var = "pressao_oleo_motor_bar"

    elif tipo == "rpm_instavel":
        for k, i in enumerate(range(start_idx, end)):
            df.at[i, "rpm_motor"] = df.at[i, "rpm_motor"] + 200.0 * ((-1) ** k)
            _set_label(df, i, "motor", "atencao")
        var = "rpm_motor"
    else:
        raise ValueError(f"tipo de anomalia de motor desconhecido: {tipo}")

    if events is not None:
        events.append({"inicio": ts_ini, "fim": ts_fim, "componente": "motor",
                       "variavel": var, "tipo": tipo})
    return df


def inject_transmissao_anomaly(df: pd.DataFrame, start_idx: int, tipo: str,
                               duration: int = 9,
                               severidade: Optional[str] = None,
                               events: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Injeta anomalia de TRANSMISSÃO a partir de `start_idx`.

    Tipos:
        'pressao_baixa' — pressao_oleo_transmissao cai abaixo de 13,8 bar
            (<200 psi). Fonte: justanswer; greentractortalk.
        'vibracao' — vibração x/y entra e permanece na Zona C/D (>2,8 mm/s),
            mínimo 45 min (9 leituras). Fonte: ISO 20816-3:2022.
        'rpm_tdp_instavel' — oscilação de ±50 rpm no engajamento da TDP.

    Se `severidade` for informado (atencao|alerta|critico) e tipo=='vibracao',
    usa a banda de vibração da severidade (BANDA_TRANS_VIB, Zonas B/C/D da
    ISO 20816-3) e força esse rótulo — usado pela escada de severidade (PASSO 2).
    """
    end = min(start_idx + duration, len(df))
    ts_ini, ts_fim = df.at[start_idx, "timestamp"], df.at[end - 1, "timestamp"]

    if tipo == "pressao_baixa":
        for i in range(start_idx, end):
            p = np.random.uniform(11.0, 13.6)
            df.at[i, "pressao_oleo_transmissao_bar"] = p
            sev = "critico" if p < PRES_TRANS_CRITICO else "alerta"
            _set_label(df, i, "transmissao", sev)
        var = "pressao_oleo_transmissao_bar"

    elif tipo == "vibracao" and severidade in BANDA_TRANS_VIB:
        lo, hi = BANDA_TRANS_VIB[severidade]
        for i in range(start_idx, end):
            vx = np.random.uniform(lo, hi)
            vy = np.random.uniform(lo, hi)
            df.at[i, "vibracao_eixo_x_mms"] = vx
            df.at[i, "vibracao_eixo_y_mms"] = vy
            df.at[i, "vibracao_eixo_z_mms"] = max(
                df.at[i, "vibracao_eixo_z_mms"], np.random.uniform(lo * 0.7, hi * 0.7))
            _set_label(df, i, "transmissao", severidade)
            if df.at[i, "rpm_tdp"] > 0:
                df.at[i, "rpm_tdp"] += np.random.normal(0.0, 30.0)
        var = "vibracao_eixo_x_mms"

    elif tipo == "vibracao":
        for i in range(start_idx, end):
            vx = np.random.uniform(3.0, 5.2)  # Zona C/D
            vy = np.random.uniform(3.0, 5.2)
            df.at[i, "vibracao_eixo_x_mms"] = vx
            df.at[i, "vibracao_eixo_y_mms"] = vy
            df.at[i, "vibracao_eixo_z_mms"] = max(df.at[i, "vibracao_eixo_z_mms"],
                                                  np.random.uniform(2.0, 3.5))
            sev = "critico" if max(vx, vy) > VIB_ZONA_C else "alerta"
            _set_label(df, i, "transmissao", sev)
            # Vibração alta desestabiliza a TDP (correlação 6 reforçada).
            if df.at[i, "rpm_tdp"] > 0:
                df.at[i, "rpm_tdp"] += np.random.normal(0.0, 30.0)
        var = "vibracao_eixo_x_mms"

    elif tipo == "rpm_tdp_instavel":
        for k, i in enumerate(range(start_idx, end)):
            if df.at[i, "rpm_tdp"] > 0:
                df.at[i, "rpm_tdp"] += 50.0 * ((-1) ** k)
                _set_label(df, i, "transmissao", "atencao")
        var = "rpm_tdp"
    else:
        raise ValueError(f"tipo de anomalia de transmissao desconhecido: {tipo}")

    if events is not None:
        events.append({"inicio": ts_ini, "fim": ts_fim, "componente": "transmissao",
                       "variavel": var, "tipo": tipo})
    return df


def inject_hidraulico_anomaly(df: pd.DataFrame, start_idx: int, tipo: str,
                              duration: int = 6,
                              severidade: Optional[str] = None,
                              events: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Injeta anomalia de HIDRÁULICO a partir de `start_idx`.

    Tipos:
        'pressao_baixa' — pressao_hidraulica cai abaixo de 150 bar em operação
            com carga (normal ~185 bar).
        'temp_alta' — temp_oleo_hidraulico > 93 °C por >20 min (>=4 leituras).
            Fonte: Manual JD OMKK25455 (SPN 1638.16).
        'vazao_cadente' — vazao_bomba cai de 114 para 85 L/min em ~3h (desgaste
            da bomba).

    Se `severidade` for informado (atencao|alerta|critico) e tipo=='temp_alta',
    usa a banda de temperatura da severidade (BANDA_HID_TEMP) e força esse
    rótulo — usado pela escada de severidade da calibração (PASSO 2).
    """
    if tipo == "vazao_cadente" and duration < 36:
        duration = 36  # ~3h (36 leituras de 5 min)
    end = min(start_idx + duration, len(df))
    ts_ini, ts_fim = df.at[start_idx, "timestamp"], df.at[end - 1, "timestamp"]

    if tipo == "pressao_baixa":
        for i in range(start_idx, end):
            p = np.random.uniform(130.0, 148.0)
            df.at[i, "pressao_hidraulica_bar"] = p
            sev = "critico" if p < PRES_HID_CRITICO else "alerta"
            _set_label(df, i, "hidraulico", sev)
        var = "pressao_hidraulica_bar"

    elif tipo == "temp_alta" and severidade in BANDA_HID_TEMP:
        lo, hi = BANDA_HID_TEMP[severidade]
        for i in range(start_idx, end):
            t = np.random.uniform(lo, hi)
            df.at[i, "temp_oleo_hidraulico_c"] = t
            df.at[i, "temp_freio_c"] = t - 3.0 + np.random.normal(0.0, 1.0)  # Hy-Gard
            _set_label(df, i, "hidraulico", severidade)
        var = "temp_oleo_hidraulico_c"

    elif tipo == "temp_alta":
        for i in range(start_idx, end):
            t = np.random.uniform(94.0, 105.0)
            df.at[i, "temp_oleo_hidraulico_c"] = t
            df.at[i, "temp_freio_c"] = t - 3.0 + np.random.normal(0.0, 1.0)  # Hy-Gard
            sev = "critico" if t > TEMP_HID_CRITICO else "alerta"
            _set_label(df, i, "hidraulico", sev)
        var = "temp_oleo_hidraulico_c"

    elif tipo == "vazao_cadente":
        for k, i in enumerate(range(start_idx, end)):
            frac = k / max(1, (end - start_idx - 1))
            v = VAZAO_PFC_BASE - (VAZAO_PFC_BASE - 85.0) * frac + np.random.normal(0.0, 1.0)
            df.at[i, "vazao_bomba_lmin"] = v
            sev = "alerta" if v < 95.0 else "atencao"
            _set_label(df, i, "hidraulico", sev)
        var = "vazao_bomba_lmin"
    else:
        raise ValueError(f"tipo de anomalia de hidraulico desconhecido: {tipo}")

    if events is not None:
        events.append({"inicio": ts_ini, "fim": ts_fim, "componente": "hidraulico",
                       "variavel": var, "tipo": tipo})
    return df


def inject_freios_anomaly(df: pd.DataFrame, start_idx: int, tipo: str,
                          duration: int = 8,
                          severidade: Optional[str] = None,
                          events: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Injeta anomalia de FREIOS a partir de `start_idx`.

    Tipos:
        'pressao_inconsistente' — pressao_freio oscila ±30% em frenagens
            consecutivas (pedal esponjoso simulado).
        'temp_alta' — temp_freio > 93 °C após sequência de frenagens em descida
            simulada.

    Se `severidade` for informado (atencao|alerta|critico) e tipo=='temp_alta',
    usa a banda de temperatura da severidade (BANDA_FREIO_TEMP) e força esse
    rótulo — usado pela escada de severidade da calibração (PASSO 2).
    """
    end = min(start_idx + duration, len(df))
    ts_ini, ts_fim = df.at[start_idx, "timestamp"], df.at[end - 1, "timestamp"]

    if tipo == "pressao_inconsistente":
        base = 30.0
        for k, i in enumerate(range(start_idx, end)):
            fator = 1.0 + 0.30 * ((-1) ** k)  # ±30%
            df.at[i, "pressao_freio_bar"] = base * fator
            df.at[i, "desaceleracao_ms2"] = np.random.uniform(1.5, 3.0)  # frenagem irregular
            _set_label(df, i, "freios", "atencao")
        var = "pressao_freio_bar"

    elif tipo == "temp_alta" and severidade in BANDA_FREIO_TEMP:
        lo, hi = BANDA_FREIO_TEMP[severidade]
        for i in range(start_idx, end):
            t = np.random.uniform(lo, hi)
            df.at[i, "temp_freio_c"] = t
            df.at[i, "pressao_freio_bar"] = np.random.uniform(PRES_FREIO_MIN, PRES_FREIO_MAX)
            df.at[i, "desaceleracao_ms2"] = np.random.uniform(DESACEL_MIN, DESACEL_MAX)
            _set_label(df, i, "freios", severidade)
        var = "temp_freio_c"

    elif tipo == "temp_alta":
        for i in range(start_idx, end):
            t = np.random.uniform(94.0, 104.0)
            df.at[i, "temp_freio_c"] = t
            df.at[i, "pressao_freio_bar"] = np.random.uniform(PRES_FREIO_MIN, PRES_FREIO_MAX)
            df.at[i, "desaceleracao_ms2"] = np.random.uniform(DESACEL_MIN, DESACEL_MAX)
            sev = "critico" if t > 100.0 else "alerta"
            _set_label(df, i, "freios", sev)
        var = "temp_freio_c"
    else:
        raise ValueError(f"tipo de anomalia de freios desconhecido: {tipo}")

    if events is not None:
        events.append({"inicio": ts_ini, "fim": ts_fim, "componente": "freios",
                       "variavel": var, "tipo": tipo})
    return df


# ===========================================================================
# DEGRADAÇÃO PROGRESSIVA (perfil antigo)
# ===========================================================================

def aplicar_degradacao_antigo(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica degradação contínua e suave ao trator antigo (manutenção atrasada).

    Semana 1 (d0-6): próximo do normal.
    Semana 2 (d7-13): temperatura do motor subindo progressivamente.
    Semana 3 (d14-20): vibração nos eixos aumentando (rolamentos desgastados).
    Semana 4 (d21-29): pressão hidráulica caindo, alarmes frequentes.

    A degradação é aditiva e cresce com o dia (sem saltos abruptos). Os rótulos
    são derivados por limiares físicos sobre as variáveis já degradadas.
    """
    d = df["_day"].to_numpy().astype(float)
    n = len(df)

    # Drift de temperatura (semana 2 em diante) — fonte: deriva térmica por
    # arrefecimento sujo/viscoso (Manual PowerTech: radiador/fan clutch).
    drift_temp = np.clip(d - 7.0, 0.0, None) / 22.0 * 13.0
    df["temp_arrefecimento_c"] = df["temp_arrefecimento_c"] + drift_temp
    df["temp_oleo_motor_c"] = df["temp_arrefecimento_c"] + TEMP_OLEO_MOTOR_OFFSET

    # Drift de vibração (semana 3 em diante) — rolamentos desgastados (ISO 20816-3).
    drift_vib = np.clip(d - 14.0, 0.0, None) / 15.0 * 2.2
    df["vibracao_eixo_x_mms"] = df["vibracao_eixo_x_mms"] + drift_vib
    df["vibracao_eixo_y_mms"] = df["vibracao_eixo_y_mms"] + drift_vib * 0.9

    # Drift de queda de pressão hidráulica (semana 4) — desgaste da bomba/vedações.
    drift_hid = np.clip(d - 21.0, 0.0, None) / 8.0 * 50.0
    campo = df["_regime"].to_numpy() == "campo"
    df.loc[campo, "pressao_hidraulica_bar"] = df.loc[campo, "pressao_hidraulica_bar"] - drift_hid[campo]

    # Rotulagem por limiares físicos sobre as variáveis degradadas.
    temp = df["temp_arrefecimento_c"].to_numpy()
    vibmax = df[["vibracao_eixo_x_mms", "vibracao_eixo_y_mms", "vibracao_eixo_z_mms"]].max(axis=1).to_numpy()
    presh = df["pressao_hidraulica_bar"].to_numpy()
    for i in range(n):
        sev = "normal"
        comp = "none"
        # Motor (temperatura)
        if temp[i] > TEMP_ARREF_CRITICO:
            sev, comp = _classificar_severidade(sev, "critico"), "motor"
        elif temp[i] > TEMP_ARREF_ALERTA and SEV_RANK["alerta"] > SEV_RANK[sev]:
            sev, comp = "alerta", "motor"
        elif temp[i] > TEMP_ARREF_ATENCAO and SEV_RANK["atencao"] > SEV_RANK[sev]:
            sev, comp = "atencao", "motor"
        # Transmissão (vibração)
        if vibmax[i] > VIB_ZONA_C and SEV_RANK["critico"] > SEV_RANK[sev]:
            sev, comp = "critico", "transmissao"
        elif vibmax[i] > VIB_ZONA_B and SEV_RANK["alerta"] > SEV_RANK[sev]:
            sev, comp = "alerta", "transmissao"
        # Hidráulico (pressão sob carga)
        if campo[i]:
            if presh[i] < PRES_HID_CRITICO and SEV_RANK["critico"] > SEV_RANK[sev]:
                sev, comp = "critico", "hidraulico"
            elif presh[i] < PRES_HID_ALERTA and SEV_RANK["alerta"] > SEV_RANK[sev]:
                sev, comp = "alerta", "hidraulico"
            elif presh[i] < PRES_HID_ATENCAO and SEV_RANK["atencao"] > SEV_RANK[sev]:
                sev, comp = "atencao", "hidraulico"
        if sev != "normal":
            df.at[i, "severidade"] = sev
            df.at[i, "componente_anomalo"] = comp
            df.at[i, "anomalia_injetada"] = 1
    return df


# ===========================================================================
# ESCADA DE SEVERIDADE (garante âncoras de calibração por componente)
# ===========================================================================

def _injetar_escada_severidade(df: pd.DataFrame, dur: int = 12,
                               frac_ini: float = 0.04, frac_fim: float = 0.96,
                               events: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Injeta, para CADA um dos 4 componentes, um evento por severidade
    (atencao/alerta/critico) — 12 eventos no total, espaçados uniformemente.

    Objetivo: garantir que TODO par (componente, severidade) exista no CSV de
    cada trator, dando à calibração por severidade (PASSO 2) âncoras de score
    bruto em normal/atencao/alerta/critico. Sem isso, modelos como a transmissão
    do trator_novo (que nunca via falha) não conseguem mapear gravidade → score.

    Cada evento tem `dur` leituras consecutivas (>=6 = contrato do buffer da
    API). As magnitudes vêm das bandas fisicamente fundamentadas BANDA_*.
    """
    n = len(df)
    plano = [(comp, sev) for comp in ("motor", "transmissao", "hidraulico", "freios")
             for sev in SEV_LADDER]
    starts = np.linspace(int(frac_ini * n), int(frac_fim * n) - dur, len(plano))
    for (comp, sev), start in zip(plano, starts.astype(int)):
        s = int(start)
        if comp == "motor":
            inject_motor_anomaly(df, s, "superaquecimento", duration=dur,
                                 severidade=sev, events=events)
        elif comp == "transmissao":
            inject_transmissao_anomaly(df, s, "vibracao", duration=dur,
                                       severidade=sev, events=events)
        elif comp == "hidraulico":
            inject_hidraulico_anomaly(df, s, "temp_alta", duration=dur,
                                      severidade=sev, events=events)
        else:
            inject_freios_anomaly(df, s, "temp_alta", duration=dur,
                                  severidade=sev, events=events)
    return df


# ===========================================================================
# 2. FUNÇÕES DE GERAÇÃO POR PERFIL
# ===========================================================================

def generate_trator_novo() -> pd.DataFrame:
    """Gera o trator_novo: manutenção em dia, operação majoritariamente saudável.

    - Variáveis na faixa normal com ruído leve na base.
    - Escada de severidade completa nos 4 componentes (atencao/alerta/critico):
      antes, o novo só tinha falha de MOTOR (atencao), então os modelos de
      transmissão/hidráulico/freios nunca viam falha e não passavam de ~74 no
      score. Agora cada componente tem exemplos rotulados de gravidade crescente,
      permitindo que a calibração por severidade mapeie crítico → ~90 (vermelho).
    - ~4% das leituras com anomalia (12 eventos x 12 leituras ≈ 144 linhas).
    - Horas acumuladas: 250 → ~700.
    """
    df = _gerar_base("novo")
    df = aplicar_correlacoes(df)
    events: List[Dict] = []
    _injetar_escada_severidade(df, dur=12, events=events)
    df.attrs["events"] = events
    return df


def generate_trator_antigo() -> pd.DataFrame:
    """Gera o trator_antigo: manutenção atrasada, degradação progressiva.

    - Degradação contínua e suave ao longo das 4 semanas.
    - ~20% das leituras com anomalia, mix de atencao/alerta/critico.
    - Críticos concentrados na semana 4.
    - Horas acumuladas: 4.200 → ~4.650.
    """
    df = _gerar_base("antigo")
    df = aplicar_correlacoes(df)
    df = aplicar_degradacao_antigo(df)
    events: List[Dict] = []

    # Eventos discretos para diversificar o mix de falhas (impacto pequeno no %).
    inject_hidraulico_anomaly(df, int(0.45 * len(df)), "temp_alta", duration=6, events=events)
    inject_transmissao_anomaly(df, int(0.60 * len(df)), "vibracao", duration=9, events=events)
    # Freios na semana 4 (>=75% do mês): temperatura alta após uso intenso.
    inject_freios_anomaly(df, int(0.85 * len(df)), "temp_alta", duration=6, events=events)
    # Escada de severidade — garante âncoras de calibração (PASSO 2) em todos os
    # (componente, severidade), inclusive os não cobertos pela degradação.
    _injetar_escada_severidade(df, dur=12, events=events)
    df.attrs["events"] = events
    return df


def generate_trator_critico() -> pd.DataFrame:
    """Gera o trator_critico: terreno adverso, picos abruptos em rajadas.

    - Picos de vibração/pressão hidráulica em clusters de 20–40 leituras.
    - Inclinação lateral frequentemente elevada (terreno irregular).
    - ~30% das leituras com anomalia, em rajadas concentradas.
    - Entre rajadas: operação relativamente normal (dificulta o modelo).
    - Horas acumuladas: 1.800 → ~2.250.
    """
    df = _gerar_base("critico")
    df = aplicar_correlacoes(df)
    events: List[Dict] = []

    dias = np.unique(df["_day"].to_numpy())
    idx_por_dia = {d: np.where(df["_day"].to_numpy() == d)[0] for d in dias}

    for d in dias:
        linhas = idx_por_dia[d]
        n_rajadas = np.random.randint(1, 3)  # 1–2 rajadas por dia operacional
        for _ in range(n_rajadas):
            L = int(np.random.randint(20, 41))  # cluster de 20–40 leituras
            if len(linhas) <= L:
                continue
            s_local = int(np.random.randint(0, len(linhas) - L))
            start_idx = int(linhas[s_local])
            comp = np.random.choice(
                ["transmissao", "hidraulico", "motor", "freios"],
                p=[0.45, 0.30, 0.15, 0.10])
            if comp == "transmissao":
                inject_transmissao_anomaly(df, start_idx, "vibracao", duration=L, events=events)
            elif comp == "hidraulico":
                # 'temp_alta' é plausível em qualquer regime (evita pressão baixa
                # incoerente durante o standby de transporte ~41 bar).
                inject_hidraulico_anomaly(df, start_idx, "temp_alta", duration=L, events=events)
            elif comp == "motor":
                inject_motor_anomaly(df, start_idx, "superaquecimento",
                                     duration=min(L, 24), peak=112.0, events=events)
            else:  # freios
                tipo = np.random.choice(["temp_alta", "pressao_inconsistente"])
                inject_freios_anomaly(df, start_idx, tipo, duration=min(L, 12), events=events)
            # Terreno adverso: inclinação elevada durante a rajada.
            fim = min(start_idx + L, len(df))
            df.loc[start_idx:fim - 1, "angulo_inclinacao_graus"] = np.abs(
                np.random.normal(15.0, 5.0, fim - start_idx))

    # Escada de severidade — garante âncoras de calibração (PASSO 2) em todos os
    # (componente, severidade), independentemente do sorteio das rajadas.
    _injetar_escada_severidade(df, dur=12, events=events)
    df.attrs["events"] = events
    return df


# ===========================================================================
# FINALIZAÇÃO (limites físicos, limpeza, ordenação de colunas)
# ===========================================================================

def _finalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limites físicos, remove auxiliares e garante zero NaN/inf.

    Clip em todas as variáveis numéricas (nada negativo, dentro de faixas
    plausíveis), substituição de qualquer inf por limite e checagem final.
    """
    for col, (lo, hi) in LIMITES_FISICOS.items():
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(lo)
        df[col] = df[col].clip(lower=lo, upper=hi)

    # Inteiros/labels coerentes.
    df["rpm_motor"] = df["rpm_motor"].round(0)
    df["rpm_tdp"] = df["rpm_tdp"].round(0)
    df["anomalia_injetada"] = df["anomalia_injetada"].astype(int)
    df["horas_uso_acumuladas"] = df["horas_uso_acumuladas"].round(1)

    # Arredondamento para leitura amigável (sem perder realismo).
    cols_2dec = [c for c in LIMITES_FISICOS if c not in ("rpm_motor", "rpm_tdp")]
    df[cols_2dec] = df[cols_2dec].round(2)

    df = df.drop(columns=[c for c in ("_day", "_regime") if c in df.columns])
    df = df[COLUNAS]

    assert not df[COLUNAS].drop(columns=["timestamp", "equipamento_id",
                                         "regime_operacao", "componente_anomalo",
                                         "severidade"]).isin([np.inf, -np.inf]).any().any()
    assert df.isna().sum().sum() == 0, "CSV final contém NaN"
    return df


# ===========================================================================
# 7. RELATÓRIO FINAL
# ===========================================================================

def _relatorio(dfs: Dict[str, pd.DataFrame]) -> None:
    """Imprime o relatório de validação do dataset gerado."""
    print("\n" + "=" * 78)
    print("AGROGUARD — RELATÓRIO DE GERAÇÃO DO DATASET (John Deere 6M)")
    print("=" * 78)

    for nome, df in dfs.items():
        total = len(df)
        anom = int((df["anomalia_injetada"] == 1).sum())
        pct = 100.0 * anom / total if total else 0.0
        print(f"\n--- {nome} ---")
        print(f"  Total de leituras : {total}")
        print(f"  Anomalias         : {anom} ({pct:.1f}%)")

        if nome in ("trator_antigo", "trator_critico"):
            print("  Anomalias por componente (sobre o total de leituras):")
            for comp in ("motor", "transmissao", "hidraulico", "freios"):
                c = int((df["componente_anomalo"] == comp).sum())
                print(f"    - {comp:<12}: {c:>5} ({100.0 * c / total:.1f}%)")
            print("  Distribuição por severidade:")
            for sev in ("atencao", "alerta", "critico"):
                c = int((df["severidade"] == sev).sum())
                print(f"    - {sev:<8}: {c:>5} ({100.0 * c / total:.1f}%)")

    # Ranges reais vs faixa esperada (dataset completo).
    completo = pd.concat(dfs.values(), ignore_index=True)
    print("\n" + "-" * 78)
    print("RANGE REAL GERADO vs FAIXA ESPERADA DO RELATÓRIO TÉCNICO")
    print("-" * 78)
    print(f"{'variavel':<30}{'min':>8}{'max':>9}{'media':>9}{'std':>8}  esperado")
    for col in FAIXAS_ESPERADAS:
        s = completo[col]
        print(f"{col:<30}{s.min():>8.1f}{s.max():>9.1f}{s.mean():>9.1f}"
              f"{s.std():>8.1f}  {FAIXAS_ESPERADAS[col]}")
    print("=" * 78)


# ===========================================================================
# 5. FUNÇÃO PRINCIPAL
# ===========================================================================

def main() -> None:
    """Orquestra a geração dos 3 perfis, aplica finalização e salva os CSVs.

    Justificativa das proporções (alvo: novo ~5% / antigo ~20% / critico ~30%):
    # Isolation Forest aprende o padrão normal. Proporção alta de dados normais
    # é intencional — garantir que o modelo aprenda a distribuição saudável,
    # não a distribuição de falha.
    """
    os.makedirs(DIR_SAIDA, exist_ok=True)

    geradores = {
        "trator_novo": generate_trator_novo,
        "trator_antigo": generate_trator_antigo,
        "trator_critico": generate_trator_critico,
    }

    dfs: Dict[str, pd.DataFrame] = {}
    for nome, gerar in geradores.items():
        df = _finalizar(gerar())
        df.to_csv(os.path.join(DIR_SAIDA, f"{nome}.csv"), index=False)
        dfs[nome] = df

    completo = pd.concat(dfs.values(), ignore_index=True)
    completo.to_csv(os.path.join(DIR_SAIDA, "dataset_completo.csv"), index=False)

    _relatorio(dfs)
    print(f"\nArquivos salvos em '{DIR_SAIDA}/': "
          "trator_novo.csv, trator_antigo.csv, trator_critico.csv, dataset_completo.csv")


if __name__ == "__main__":
    main()
