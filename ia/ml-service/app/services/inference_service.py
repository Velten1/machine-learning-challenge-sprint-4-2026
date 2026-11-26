"""Régua ÚNICA de risco: z-magnitude → score 0–100 → cor/nível/alerta.

Uma decisão, não quatro. O score vem do z-magnitude (maior desvio robusto entre
os sensores do componente, condicionado ao regime) calibrado por severidade no
treino. A MESMA grandeza alimenta o texto de explicação, então score e texto
nunca mais divergem. O Isolation Forest entra só como GATE de novidade (pega
anomalias de padrão/oscilação) e há um piso físico para limites absolutos.
"""
from __future__ import annotations

import numpy as np

COMPONENTES = ["motor", "transmissao", "hidraulico", "freios"]

PISO_CRITICO = 81.0  # piso de score quando um limite físico absoluto é violado

# Limites físicos absolutos (violação = crítico inequívoco), com fonte:
# - temp_arrefecimento > 112 °C: parada recomendada do motor [OFICIAL PowerTech].
# - pressao_oleo_motor < 0,8 bar (com motor ligado): perda crítica de pressão
#   (mínimo de marcha lenta 1,03 bar) [OFICIAL PowerTech].
# - temp_oleo_motor > 128 °C: superaquecimento severo do óleo.
# - vibração > 4,5 mm/s: Zona D [ISO 20816-3:2022] — dano iminente.
# - pressao_oleo_transmissao < 12 bar: DTC crítico (<200 psi severo).
# - temp_oleo_hidraulico > 108 °C: superaquecimento severo (SPN 1638.16).
# - temp_freio > 110 °C: fade térmico severo dos discos.


def _g(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def excede_limite_fisico(comp: str, row: dict) -> bool:
    """True se algum sensor do componente violar um limite físico absoluto."""
    if comp == "motor":
        return (_g(row, "temp_arrefecimento_c") > 112.0
                or 0.0 < _g(row, "pressao_oleo_motor_bar") < 0.8
                or _g(row, "temp_oleo_motor_c") > 128.0)
    if comp == "transmissao":
        vib = max(_g(row, "vibracao_eixo_x_mms"),
                  _g(row, "vibracao_eixo_y_mms"),
                  _g(row, "vibracao_eixo_z_mms"))
        return vib > 4.5 or 0.0 < _g(row, "pressao_oleo_transmissao_bar") < 12.0
    if comp == "hidraulico":
        return _g(row, "temp_oleo_hidraulico_c") > 108.0
    if comp == "freios":
        return _g(row, "temp_freio_c") > 110.0
    return False


def score_to_level(score: float) -> dict:
    """Régua única score → cor / nível / se deve explicar.

    verde 0–30 normal | amarelo 31–60 atencao | laranja 61–80 alto |
    vermelho 81–100 critico.
    """
    if score <= 30:
        return {"cor": "verde", "nivel": "normal", "explicar": False}
    if score <= 60:
        return {"cor": "amarelo", "nivel": "atencao", "explicar": True}
    if score <= 80:
        return {"cor": "laranja", "nivel": "alto", "explicar": True}
    return {"cor": "vermelho", "nivel": "critico", "explicar": True}


def cor_score(score: float) -> str:
    return score_to_level(score)["cor"]


def z_componente(row: dict, base: list[str], baselines: dict) -> float:
    """Maior |x - média| / desvio entre os sensores, baseline por regime da leitura."""
    regime = str(row.get("regime_operacao", "global"))
    z = 0.0
    for c in base:
        bl = baselines[c]
        ref = bl.get(regime, bl["global"])
        s = ref["std"] or 1e-6
        z = max(z, abs((_g(row, c) - ref["mean"]) / s))
    return z


def para_score(z: float, raw_anchors: list[float],
               score_anchors: list[float]) -> float:
    """z calibrado → 0–100 (np.interp satura: <piso → 0, >topo → 100)."""
    return float(np.interp(z, raw_anchors, score_anchors))


def calibrar_componentes(
    row: dict,
    raw_if: dict[str, float],
    meta: dict,
) -> dict[str, dict]:
    """Converte a última leitura em score único, cor, nível e threshold por componente."""
    out: dict[str, dict] = {}
    for comp in COMPONENTES:
        cfg = meta["componentes"][comp]
        cal = cfg["calibracao_severidade"]
        base = cfg["base_features"]
        baselines = cfg["baselines"]
        threshold = float(cfg.get("threshold", 31.0))

        z = z_componente(row, base, baselines)
        score = para_score(z, cal["raw_anchors"], cal["score_anchors"])

        # Gate de novidade do IF: anomalia de padrão que o z de valor não pegou
        # eleva ao mínimo de alerta (amarelo), sem deixar o IF achatar o topo.
        if raw_if.get(comp, 0.0) > float(cfg.get("if_gate", np.inf)) and score < threshold:
            score = threshold

        # Piso físico: limite absoluto violado → vermelho garantido.
        if excede_limite_fisico(comp, row):
            score = max(score, PISO_CRITICO)

        nivel = score_to_level(score)
        out[comp] = {
            "score": round(score, 1),
            "cor": nivel["cor"],
            "nivel": nivel["nivel"],
            "explicar": nivel["explicar"],
            "threshold": round(threshold, 1),
        }
    return out
