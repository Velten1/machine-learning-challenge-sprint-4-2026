"""Explicações textuais alinhadas à régua única (z-magnitude por regime).

O texto usa a MESMA grandeza do score (z robusto por regime), então score, cor e
texto contam a mesma história. Só explica quando a régua manda (explicar=True,
score >= 31); o nível do texto vem do nível da cor; o % tem teto de 200%.
"""
from __future__ import annotations

import pandas as pd

ACAO = {
  "motor": "reduzir carga e inspecionar o sistema de arrefecimento",
  "transmissao": "verificar óleo Hy-Gard e a vibração dos eixos",
  "hidraulico": "inspecionar bomba, filtros e temperatura do óleo",
  "freios": "verificar discos molhados e a pressão de acionamento",
}

COMPONENTE_NOME = {
  "motor": "Motor",
  "transmissao": "Transmissão",
  "hidraulico": "Sistema hidráulico",
  "freios": "Freios",
}

LABEL_PT = {
  "temp_arrefecimento_c": "temperatura de arrefecimento",
  "rpm_motor": "RPM do motor",
  "pressao_oleo_motor_bar": "pressão do óleo do motor",
  "temp_oleo_motor_c": "temperatura do óleo do motor",
  "consumo_combustivel_lh": "consumo de combustível",
  "pressao_oleo_transmissao_bar": "pressão do óleo da transmissão",
  "rpm_tdp": "RPM da TDP",
  "vibracao_eixo_x_mms": "vibração no eixo X",
  "vibracao_eixo_y_mms": "vibração no eixo Y",
  "vibracao_eixo_z_mms": "vibração no eixo Z",
  "pressao_hidraulica_bar": "pressão hidráulica",
  "temp_oleo_hidraulico_c": "temperatura do óleo hidráulico",
  "vazao_bomba_lmin": "vazão da bomba",
  "pressao_freio_bar": "pressão do freio",
  "desaceleracao_ms2": "desaceleração",
  "temp_freio_c": "temperatura dos freios",
}

PCT_MAX = 200.0  # teto do desvio relatado (acaba com "392%"/"999%")


def _vars_texto(row: pd.Series, base: list[str], baselines: dict,
                regime: str, top_n: int = 3) -> str:
    zs = {}
    for c in base:
        ref = baselines[c].get(regime, baselines[c]["global"])
        s = ref["std"] or 1e-6
        zs[c] = (float(row[c]) - ref["mean"]) / s
    top = sorted(zs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    partes = []
    for var, z in top:
        pct = min(abs(z) * 100.0 / 3.0, PCT_MAX)
        prefixo = "mais de " if abs(z) * 100.0 / 3.0 > PCT_MAX else ""
        direcao = "acima" if z >= 0 else "abaixo"
        partes.append(f"{LABEL_PT.get(var, var)} {prefixo}{pct:.0f}% {direcao} do padrão")
    return "; ".join(partes)


def explicar(comp: str, row: pd.Series, baselines: dict, base: list[str],
             nivel: str, regime: str) -> str:
    nome = COMPONENTE_NOME.get(comp, comp)
    vars_txt = _vars_texto(row, base, baselines, regime)
    if nivel == "atencao":
        return f"Atenção: {nome} — {vars_txt}."
    if nivel == "alto":
        return f"Risco alto: {nome} — {vars_txt}. Recomendação: {ACAO[comp]}."
    # critico
    return (f"Crítico: {nome} — {vars_txt}. Interromper a operação e "
            f"inspecionar imediatamente ({ACAO[comp]}).")


def anexar_explicacoes(
  df: pd.DataFrame,
  meta: dict,
  componentes: dict[str, dict],
) -> dict[str, dict]:
  """Gera explicação apenas quando a régua manda (explicar=True / score >= 31)."""
  linha = df.iloc[-1]
  regime = str(linha.get("regime_operacao", "global"))
  for comp, data in componentes.items():
    if data.get("explicar"):
      cfg = meta["componentes"][comp]
      data["explicacao"] = explicar(
        comp, linha, cfg["baselines"], cfg["base_features"],
        data["nivel"], regime,
      )
    else:
      data["explicacao"] = None
  return componentes
