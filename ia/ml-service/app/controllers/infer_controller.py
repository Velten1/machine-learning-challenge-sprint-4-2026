"""Orquestra inferência + explicação e monta resposta final."""
from __future__ import annotations

from engine import InferenceEngine
from app.services.explanation_service import anexar_explicacoes
from app.services.inference_service import calibrar_componentes, COMPONENTES
from app.schemas.telemetria import InferRequest, InferResponse, ComponentScoreOut

# Perfil narrativo de cada equipamento; usado como fallback quando o
# metadata.json não traz o campo "perfil".
PERFIS_FALLBACK = {
  "trator_novo": "saudavel",
  "trator_antigo": "degradado",
  "trator_critico": "hostil",
  "demo_novo": "saudavel",
}


class InferController:
  def __init__(self, engine: InferenceEngine) -> None:
    self.engine = engine

  def inferir(self, body: InferRequest) -> InferResponse:
    buffer = [item.model_dump() for item in body.buffer[-6:]]
    equipamento_id = body.equipamento_id
    meta = self.engine.get_metadata(equipamento_id)

    df, raw_if = self.engine.raw_scores(equipamento_id, buffer)
    row = df.iloc[-1].to_dict()
    componentes = calibrar_componentes(row, raw_if, meta)
    componentes = anexar_explicacoes(df, meta, componentes)

    max_score = -1.0
    componente_critico = COMPONENTES[0]
    comp_out: dict[str, ComponentScoreOut] = {}

    for comp in COMPONENTES:
      c = componentes[comp]
      comp_out[comp] = ComponentScoreOut(**c)
      if c["score"] > max_score:
        max_score = c["score"]
        componente_critico = comp

    timestamp = str(df.iloc[-1].get("timestamp", ""))
    perfil = meta.get("perfil", PERFIS_FALLBACK.get(equipamento_id, "desconhecido"))

    return InferResponse(
      equipamento_id=equipamento_id,
      perfil=perfil,
      timestamp=timestamp,
      risco_geral=round(max_score, 1),
      componente_critico=componente_critico,
      buffer_size=len(buffer),
      confiavel=len(buffer) >= int(meta.get("roll_window", 6)),
      componentes=comp_out,
    )
