"""Confere os exemplos com a API original em um servidor HTTP local."""

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8885"


def main():
    cases = json.loads(
        (ROOT / "evidencias/casos-inferencia.json").read_text(encoding="utf8")
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8885",
        ],
        cwd=ROOT / "ia/ml-service",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health = None
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError(
                    "O servidor encerrou antes de iniciar. Confira a porta 8885."
                )
            try:
                with urlopen(BASE_URL + "/health", timeout=1) as response:
                    health = json.load(response)
                break
            except URLError:
                time.sleep(0.2)
        if health is None:
            raise RuntimeError("O servidor não respondeu ao health check.")

        results = []
        for case in cases:
            request = Request(
                BASE_URL + "/infer",
                data=json.dumps(case["requisicao"]).encode("utf8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=10) as response:
                result = json.load(response)
                assert result == case["resposta"], case["caso"]
                results.append(
                    {
                        "caso": case["caso"],
                        "http": response.status,
                        "risco_geral": result["risco_geral"],
                        "confiavel": result["confiavel"],
                    }
                )

        evidence = {
            "servidor": "uvicorn local",
            "porta": 8885,
            "health": {
                "status": health["status"],
                "equipamentos": health["equipamentos"],
            },
            "casos": results,
        }
        (ROOT / "evidencias/teste-http.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
        )
        print("Seis inferências HTTP conferidas com sucesso.")
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
