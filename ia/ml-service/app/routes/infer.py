"""Rotas FastAPI — apenas declara endpoints e delega ao controller."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.controllers.infer_controller import InferController
from app.schemas.telemetria import InferRequest, InferResponse

router = APIRouter()


def create_infer_router(controller: InferController) -> APIRouter:
  @router.post("/infer", response_model=InferResponse)
  def inferir(body: InferRequest) -> InferResponse:
    try:
      return controller.inferir(body)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
      raise HTTPException(status_code=500, detail=str(exc)) from exc

  return router
