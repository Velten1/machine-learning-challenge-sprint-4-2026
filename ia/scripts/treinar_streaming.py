"""Retreina 12 modelos IF com features streaming + calibração por severidade.

Régua única (uma decisão, não quatro) — alvo do produto:
    verde   0–30  : normal      (âncora p50 do normal  → 15)
    amarelo 31–60 : atencao      (âncora p50 atencao    → 40)
    laranja 61–80 : alto/alerta  (âncora p50 alerta     → 70)
    vermelho 81–100: critico     (âncora p50 critico    → 90; p95 → 100)

A calibração mapeia o score bruto do Isolation Forest para 0–100 via interpolação
linear ancorada nos percentis por severidade — não há mais "esticar p_max para
deixar normal verde" (era o que travava anomalia grave em 60–74).

Uso (na pasta ia/):
    python scripts/treinar_streaming.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml-service"))

from engine import COMPONENTES, EQUIPAMENTOS, engenharia_features  # noqa: E402


# engenharia_features(df) reproduz, linha a linha, o mesmo vetor que a API monta
# com o buffer de `window` leituras — treino e produção usam o MESMO formato.
def features_treino(df: pd.DataFrame, base: List[str], feature_cols: List[str],
                    window: int) -> pd.DataFrame:
    return engenharia_features(df, base, window)[feature_cols]


SEED = 42
ROLL_WINDOW = 6
THRESHOLD_ALERTA = 31.0          # fronteira verde/amarelo (régua única)
MIN_AMOSTRAS_SEVERIDADE = 5      # mínimo p/ usar uma severidade como âncora
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Âncoras de score final (centro de cada faixa da régua) por severidade.
ANCORA_SCORE: Dict[str, float] = {
    "normal_lo": 0.0,    # p5  do normal  → piso
    "normal": 15.0,      # p50 do normal  → centro do verde
    "atencao": 40.0,     # p50 atencao    → centro do amarelo
    "alerta": 70.0,      # p50 alerta     → centro do laranja
    "critico": 90.0,     # p50 critico    → centro do vermelho
    "critico_hi": 100.0,  # p95 critico    → topo da escala
}

BASE_FEATURES: Dict[str, List[str]] = {
    "motor": [
        "temp_arrefecimento_c", "rpm_motor", "pressao_oleo_motor_bar",
        "temp_oleo_motor_c", "consumo_combustivel_lh",
    ],
    "transmissao": [
        "pressao_oleo_transmissao_bar", "rpm_tdp",
        "vibracao_eixo_x_mms", "vibracao_eixo_y_mms", "vibracao_eixo_z_mms",
    ],
    "hidraulico": [
        "pressao_hidraulica_bar", "temp_oleo_hidraulico_c", "vazao_bomba_lmin",
    ],
    "freios": [
        "pressao_freio_bar", "desaceleracao_ms2", "temp_freio_c",
    ],
}

PERFIS = {
    "trator_novo": "saudavel",
    "trator_antigo": "degradado",
    "trator_critico": "hostil",
}


def lista_features(base: List[str]) -> List[str]:
    cols: List[str] = []
    for col in base:
        cols += [col, f"{col}_roll_std", f"{col}_delta"]
    for r in ["campo", "transporte", "ralenti", "parado"]:
        cols.append(f"regime_{r}")
    return cols


def construir_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("iforest", IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination="auto",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def _monotonico(pares: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    """Ordena por score-alvo e força o raw a ser estritamente crescente.

    np.interp exige x (raw) crescente; e a régua exige score crescente com o raw.
    Quando o raw de uma severidade não vem acima do anterior (ruído), é elevado
    em epsilon — degrau colapsa mas a função permanece monotônica e válida.
    """
    pares = sorted(pares, key=lambda p: p[1])  # por score-alvo (severidade)
    raws: List[float] = []
    scores: List[float] = []
    prev = -np.inf
    for raw, score in pares:
        if raw <= prev:
            raw = prev + 1e-6
        raws.append(float(raw))
        scores.append(float(score))
        prev = raw
    return raws, scores


REGIMES_BL = ["campo", "transporte", "ralenti", "parado"]


def construir_baselines(df: pd.DataFrame, base: List[str],
                        mask_normal: np.ndarray) -> Dict[str, dict]:
    """Baseline NORMAL por regime (média/desvio) + fallback global.

    Variáveis como pressão hidráulica e RPM da TDP variam muito por regime
    (campo 185 bar vs transporte 41 bar). Um desvio global achataria anomalias
    reais; condicionar ao regime dá um z fiel. O desvio de cada regime tem piso
    de 25% do desvio global para não explodir z em regimes quase constantes.
    """
    def robusto(vals: np.ndarray) -> Tuple[float, float]:
        """Centro e escala robustos por tipo de variável.

        - Contínua: mediana + 1.4826*MAD. MAD é imune à cauda de degradação
          progressiva (ex.: pressão hidráulica do trator antigo caindo aos poucos
          em leituras ainda rotuladas normais), que inflaria o desvio e mascararia
          anomalias reais.
        - Esparsa/evento (mediana e MAD ~0, como pressão de freio que fica em 0 na
          maior parte do tempo): MAD=0 estouraria o z. Usa a amplitude do normal
          (p98) como escala, de modo que um EVENTO normal (uma frenagem rotineira)
          fique perto da fronteira verde e não dispare alarme.
        """
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = 1.4826 * mad
        if scale <= 1e-9:
            p98 = float(np.percentile(vals, 98))
            scale = max((p98 - med) / 2.7, float(vals.std()), 1e-6)
        return med, float(scale)

    bl: Dict[str, dict] = {}
    ndf = df.loc[mask_normal]
    for c in base:
        g_mean, g_std = robusto(ndf[c].to_numpy(dtype=float))
        g_std = g_std or 1e-6
        piso = max(0.25 * g_std, 1e-6)
        d: Dict[str, dict] = {"global": {"mean": g_mean, "std": g_std}}
        for r in REGIMES_BL:
            sub = ndf.loc[ndf["regime_operacao"] == r, c].to_numpy(dtype=float)
            if len(sub) >= 20:
                m, s = robusto(sub)
                d[r] = {"mean": m, "std": float(max(s, piso))}
        bl[c] = d
    return bl


def z_magnitude(df: pd.DataFrame, base: List[str],
                baselines: Dict[str, dict]) -> np.ndarray:
    """Magnitude robusta da anomalia = maior |x - média| / desvio (z) entre os
    sensores do componente, com baseline NORMAL condicionada ao regime da leitura.

    Diferente do score bruto do Isolation Forest (que satura logo após a fronteira
    de outlier e não ranqueia gravidade), o z cresce de forma monotônica com a
    magnitude do desvio — é o que permite critico → ~90/100. É também a MESMA
    grandeza usada no texto de explicação (%), unificando score e texto.
    """
    n = len(df)
    reg = df["regime_operacao"].to_numpy() if "regime_operacao" in df.columns \
        else np.array(["global"] * n)
    z = np.zeros(n)
    for c in base:
        bl = baselines[c]
        vals = df[c].to_numpy(dtype=float)
        m_arr = np.full(n, bl["global"]["mean"])
        s_arr = np.full(n, bl["global"]["std"])
        for r in REGIMES_BL:
            if r in bl:
                m = reg == r
                m_arr[m] = bl[r]["mean"]
                s_arr[m] = bl[r]["std"]
        s_arr = np.where(s_arr <= 0, 1e-6, s_arr)
        z = np.maximum(z, np.abs((vals - m_arr) / s_arr))
    return z


def calibracao_severidade(sinal: np.ndarray, df: pd.DataFrame,
                          comp: str) -> Tuple[List[float], List[float]]:
    """Constrói as âncoras (sinal → score 0–100) a partir dos percentis por severidade.

    normal: p5 → 0, p50 → 15. atencao/alerta/critico: p50 → 40/70/90.
    critico: p95 → 100 (garante que anomalia grave atinja o vermelho).
    Severidades com poucas amostras são ignoradas (a interpolação usa as demais).
    """
    mask_normal = (df["anomalia_injetada"] == 0).to_numpy()
    s_norm = sinal[mask_normal]
    pares: List[Tuple[float, float]] = [
        (float(np.percentile(s_norm, 5)), ANCORA_SCORE["normal_lo"]),
        (float(np.percentile(s_norm, 50)), ANCORA_SCORE["normal"]),
    ]
    for sev in ("atencao", "alerta", "critico"):
        m = ((df["componente_anomalo"] == comp) & (df["severidade"] == sev)).to_numpy()
        if m.sum() >= MIN_AMOSTRAS_SEVERIDADE:
            pares.append((float(np.percentile(sinal[m], 50)), ANCORA_SCORE[sev]))
    m_crit = ((df["componente_anomalo"] == comp) & (df["severidade"] == "critico")).to_numpy()
    if m_crit.sum() >= MIN_AMOSTRAS_SEVERIDADE:
        pares.append((float(np.percentile(sinal[m_crit], 95)), ANCORA_SCORE["critico_hi"]))
    return _monotonico(pares)


def para_score(raw: float, raw_anchors: List[float],
               score_anchors: List[float]) -> float:
    """Mapeia score bruto → 0–100 por interpolação linear ancorada (np.interp
    satura nos extremos: <piso → 0, >topo → 100)."""
    return float(np.interp(raw, raw_anchors, score_anchors))


def treinar_equipamento(df: pd.DataFrame) -> Dict[str, dict]:
    mask_normal = (df["anomalia_injetada"] == 0).to_numpy()
    # Igual à API: treina o "normal" só com leituras de buffer cheio.
    mask_confiavel = np.arange(len(df)) >= (ROLL_WINDOW - 1)
    mask_treino = mask_normal & mask_confiavel
    artefatos: Dict[str, dict] = {}

    for comp in COMPONENTES:
        base = BASE_FEATURES[comp]
        feature_cols = lista_features(base)
        X_full = features_treino(df, base, feature_cols, ROLL_WINDOW)

        pipe = construir_pipeline()
        pipe.fit(X_full[mask_treino])

        # IF mantido como GATE de novidade (pega anomalias de padrão/oscilação
        # que o z de valor não enxerga). if_gate = p99 do IF nas normais.
        raw_if = -pipe.score_samples(X_full)
        if_gate = float(np.percentile(raw_if[mask_normal], 99))

        baselines = construir_baselines(df, base, mask_normal)

        # Régua única: calibração por severidade sobre o z-magnitude.
        z_all = z_magnitude(df, base, baselines)
        raw_anchors, score_anchors = calibracao_severidade(z_all, df, comp)

        artefatos[comp] = {
            "pipeline": pipe,
            "raw_anchors": raw_anchors,
            "score_anchors": score_anchors,
            "baselines": baselines,
            "base_features": base,
            "feature_cols": feature_cols,
            "if_gate": if_gate,
        }
    return artefatos


def salvar(eq: str, comps: Dict[str, dict]) -> None:
    out = MODELS_DIR / eq
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "equipamento_id": eq,
        "perfil": PERFIS.get(eq, "desconhecido"),
        "seed": SEED,
        "roll_window": ROLL_WINDOW,
        "regimes": ["campo", "transporte", "ralenti", "parado"],
        "score_faixas": {
            "verde": [0, 30],
            "amarelo": [31, 60],
            "laranja": [61, 80],
            "vermelho": [81, 100],
        },
        "componentes": {},
    }
    for comp, art in comps.items():
        joblib.dump(art["pipeline"], out / f"{comp}.joblib")
        meta["componentes"][comp] = {
            "base_features": art["base_features"],
            "feature_cols": art["feature_cols"],
            "calibracao_severidade": {
                "raw_anchors": art["raw_anchors"],
                "score_anchors": art["score_anchors"],
            },
            "threshold": THRESHOLD_ALERTA,
            "if_gate": art["if_gate"],
            "baselines": art["baselines"],
        }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _log(msg: str) -> None:
    print(msg, flush=True)


def validar_pos_treino() -> bool:
    """Régua única na prática:
    - leituras NORMAIS (buffer cheio) → maioria verde (score <= 30)
    - leituras CRITICO por componente → score alto (mediana >= 80, vermelho)
    """
    erros = 0
    for eq in EQUIPAMENTOS:
        meta = json.loads((MODELS_DIR / eq / "metadata.json").read_text(encoding="utf-8"))
        df = pd.read_csv(DATA_DIR / f"{eq}.csv")
        mask_normal = (df["anomalia_injetada"] == 0).to_numpy()
        mask_conf = np.arange(len(df)) >= (ROLL_WINDOW - 1)

        for comp in COMPONENTES:
            cfg = meta["componentes"][comp]
            base = cfg["base_features"]
            cal = cfg["calibracao_severidade"]
            z = z_magnitude(df, base, cfg["baselines"])
            scores = np.interp(z, cal["raw_anchors"], cal["score_anchors"])

            s_norm = scores[mask_normal & mask_conf]
            pct_verde = float((s_norm <= 30).mean()) * 100.0
            if pct_verde < 80.0:
                _log(f"  FAIL {eq}/{comp}: normais verdes {pct_verde:.0f}% (<80%)")
                erros += 1

            m_crit = ((df["componente_anomalo"] == comp) &
                      (df["severidade"] == "critico")).to_numpy()
            if m_crit.sum() >= MIN_AMOSTRAS_SEVERIDADE:
                med = float(np.median(scores[m_crit]))
                if med < 80.0:
                    _log(f"  FAIL {eq}/{comp}: critico mediana {med:.0f} (<80)")
                    erros += 1
    _log(f"Validacao: {'OK' if erros == 0 else f'{erros} falhas'}")
    return erros == 0


def main() -> None:
    for eq in EQUIPAMENTOS:
        _log(f"Treinando {eq} (streaming + severidade)...")
        df = pd.read_csv(DATA_DIR / f"{eq}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        artefatos = treinar_equipamento(df)
        salvar(eq, artefatos)
        for comp in COMPONENTES:
            art = artefatos[comp]
            ra, sa = art["raw_anchors"], art["score_anchors"]
            ancoras = ", ".join(f"{s:.0f}@{r:.2f}" for r, s in zip(ra, sa))
            _log(f"  {comp:12} ancoras[{ancoras}]")

    _log("\nValidacao pos-treino...")
    if not validar_pos_treino():
        sys.exit(1)
    _log("Concluido.")


if __name__ == "__main__":
    main()
