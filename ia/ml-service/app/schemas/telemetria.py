"""Schemas Pydantic — entrada e saída da inferência."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

REGIMES = Literal["campo", "transporte", "ralenti", "parado"]


class LeituraBuffer(BaseModel):
  timestamp: str
  equipamento_id: Optional[str] = None
  regime_operacao: REGIMES
  temp_arrefecimento_c: float = Field(..., ge=0)
  rpm_motor: float = Field(..., ge=0)
  pressao_oleo_motor_bar: float = Field(..., ge=0)
  temp_oleo_motor_c: float = Field(..., ge=0)
  consumo_combustivel_lh: float = Field(..., ge=0)
  pressao_oleo_transmissao_bar: float = Field(..., ge=0)
  rpm_tdp: float = Field(..., ge=0)
  vibracao_eixo_x_mms: float = Field(..., ge=0)
  vibracao_eixo_y_mms: float = Field(..., ge=0)
  vibracao_eixo_z_mms: float = Field(..., ge=0)
  pressao_hidraulica_bar: float = Field(..., ge=0)
  temp_oleo_hidraulico_c: float = Field(..., ge=0)
  vazao_bomba_lmin: float = Field(..., ge=0)
  pressao_freio_bar: float = Field(..., ge=0)
  desaceleracao_ms2: float = Field(..., ge=0)
  temp_freio_c: float = Field(..., ge=0)

  model_config = {"extra": "allow"}


class InferRequest(BaseModel):
  equipamento_id: str = Field(..., examples=["trator_antigo"])
  buffer: List[LeituraBuffer] = Field(..., min_length=1, max_length=6)


class ComponentScoreOut(BaseModel):
  score: float
  cor: str
  threshold: float
  explicacao: Optional[str] = None


class InferResponse(BaseModel):
  equipamento_id: str
  perfil: str
  timestamp: str
  risco_geral: float
  componente_critico: str
  buffer_size: int
  confiavel: bool
  componentes: Dict[str, ComponentScoreOut]
