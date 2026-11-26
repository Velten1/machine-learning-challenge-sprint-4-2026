"""Reproduz a verificação dos artefatos existentes, sem treinamento."""

from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ia/ml-service"))
from main import app, engine
from app.schemas.telemetria import LeituraBuffer
from app.services.inference_service import calibrar_componentes
from engine import COMPONENTES, EQUIPAMENTOS, engenharia_features, features_streaming
from fastapi.testclient import TestClient

OUT = ROOT / "evidencias"
OUT.mkdir(exist_ok=True)
client = TestClient(app)
fields = set(LeituraBuffer.model_fields)


def payload(eq, df):
    return {
        "equipamento_id": eq,
        "buffer": [
            {k: v for k, v in row.items() if k in fields}
            for row in df.to_dict("records")
        ],
    }


def main():
    summary = []
    metrics = []
    cases = []
    checks = []
    assert client.get("/health").status_code == 200
    checks.append(
        {
            "teste": "GET /health e carregamento dos 12 artefatos",
            "resultado": "aprovado",
        }
    )
    for eq in EQUIPAMENTOS:
        df = pd.read_csv(ROOT / "ia/data" / (eq + ".csv"))
        meta = engine.get_metadata(eq)
        nums = df.select_dtypes(include=np.number)
        row = {
            "equipamento": eq,
            "registros": len(df),
            "colunas": len(df.columns),
            "normais": int(df.anomalia_injetada.eq(0).sum()),
            "anomalias": int(df.anomalia_injetada.sum()),
            "nulos": int(df.isna().sum().sum()),
            "infinitos": int(np.isinf(nums).sum().sum()),
            "duplicatas": int(df.duplicated(["timestamp", "equipamento_id"]).sum()),
            "inicio": df.timestamp.min(),
            "fim": df.timestamp.max(),
        }
        summary.append(row)
        assert row["nulos"] == row["infinitos"] == row["duplicatas"] == 0
        raw = {}
        for comp in COMPONENTES:
            cfg = meta["componentes"][comp]
            X = engenharia_features(df, cfg["base_features"], 6)[cfg["feature_cols"]]
            assert not any(
                (
                    x in cfg["feature_cols"]
                    for x in ["anomalia_injetada", "componente_anomalo", "severidade"]
                )
            )
            small = features_streaming(
                df.head(16), cfg["base_features"], cfg["feature_cols"], 6
            )
            assert np.allclose(small.to_numpy(), X.head(16).to_numpy())
            raw[comp] = -engine.pipelines[eq][comp].score_samples(X)
        scores = {comp: [] for comp in COMPONENTES}
        for i, row in enumerate(df.to_dict("records")):
            cs = calibrar_componentes(
                row, {c: float(raw[c][i]) for c in COMPONENTES}, meta
            )
            for c in COMPONENTES:
                scores[c].append(cs[c]["score"])
        for comp in COMPONENTES:
            y = (
                df.anomalia_injetada.eq(1) & df.componente_anomalo.eq(comp)
            ).to_numpy()[5:]
            s = np.asarray(scores[comp])[5:]
            pred = s >= 31
            tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
            metrics.append(
                {
                    "equipamento": eq,
                    "componente": comp,
                    "n": len(y),
                    "positivos": int(y.sum()),
                    "precisao": float(precision_score(y, pred, zero_division=0)),
                    "recall": float(recall_score(y, pred, zero_division=0)),
                    "f1": float(f1_score(y, pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y, s)) if y.any() else None,
                    "AP": float(average_precision_score(y, s)) if y.any() else None,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                }
            )
        for name, idx in [
            (
                "normal",
                int(np.flatnonzero(df.anomalia_injetada.eq(0).to_numpy()[5:])[0]) + 5,
            ),
            (
                "anomalia",
                int(np.flatnonzero(df.anomalia_injetada.eq(1).to_numpy()[5:])[0]) + 5,
            ),
        ]:
            req = payload(eq, df.iloc[max(0, idx - 5) : idx + 1])
            start = time.perf_counter()
            resp = client.post("/infer", json=req)
            ms = 1000 * (time.perf_counter() - start)
            assert resp.status_code == 200
            out = resp.json()
            assert out["confiavel"] is True
            for comp in COMPONENTES:
                assert out["componentes"][comp]["score"] == scores[comp][idx]
            cases.append(
                {
                    "caso": eq + "_" + name,
                    "requisicao": req,
                    "resposta": out,
                    "tempo_local_ms": round(ms, 2),
                }
            )
        req = payload(eq, df.head(1))
        out = client.post("/infer", json=req)
        assert out.status_code == 200 and out.json()["confiavel"] is False
        checks.append(
            {
                "teste": eq
                + " dados, equivalência batch/streaming, inferência completa e warm-up",
                "resultado": "aprovado",
            }
        )
    req = payload("trator_novo", pd.read_csv(ROOT / "ia/data/trator_novo.csv").head(6))
    for name, body, status in [
        ("buffer vazio", {**req, "buffer": []}, 422),
        ("perfil inexistente", {**req, "equipamento_id": "inexistente"}, 400),
        (
            "mais de seis leituras",
            {**req, "buffer": req["buffer"] + [req["buffer"][0]]},
            422,
        ),
    ]:
        assert client.post("/infer", json=body).status_code == status
        checks.append({"teste": name, "resultado": "aprovado", "http": status})
    bad = json.loads(json.dumps(req))
    bad["buffer"][0]["rpm_motor"] = -1
    assert client.post("/infer", json=bad).status_code == 422
    checks.append(
        {"teste": "RPM negativo rejeitado", "resultado": "aprovado", "http": 422}
    )
    bad = json.loads(json.dumps(req))
    bad["buffer"][-1]["timestamp"] = "data-invalida"
    timestamp_status = client.post("/infer", json=bad).status_code
    result = {
        "origem": "artefatos preexistentes do AgroGuard",
        "commit_origem": "f7175434f2dab4f3109f832a83515b0d9a54e599",
        "escopo": "checagem retrospectiva na base sintetica de desenvolvimento, sem holdout independente",
        "treinamento_nesta_verificacao": False,
        "observacao_timestamp": {
            "http": timestamp_status,
            "limite": "o schema recebe timestamp como string sem validacao semantica estrita",
        },
        "dados": summary,
        "verificacoes": checks,
    }
    (OUT / "verificacao.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    (OUT / "casos-inferencia.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    pd.DataFrame(metrics).to_csv(OUT / "metricas-artefatos-atuais.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(pd.DataFrame(metrics).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
