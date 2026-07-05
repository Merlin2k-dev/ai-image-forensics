"""Single-image prediction entry point.

Loads the frozen model bundle and returns a verdict plus per-channel evidence
panels for one image. Pipeline per upload:

  decode -> reject if short side < 512 (never upscale)
  -> center 512 crop (plus 4 corner crops when short side >= 1024, median of 5)
  -> JPEG q75 4:2:0 re-encode (training substrate) -> 27 features
  -> channel scores -> right-tail p-value against a pooled real-photo reference
  -> banded verdict (AI / leaning AI / inconclusive / real); thresholds from the bundle's real-photo null
  -> provenance metadata reported as a separate panel, never mixed into the score

Score strength is reported as a real-photo percentile rather than a probability;
probability calibration did not transfer across corpora in testing.
"""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from PIL import Image
from scipy.stats import norm

from pipeline.features.inversion_residual import extract_inversion_features
from pipeline.features.self_consistency import extract_selfconsistency_features
from pipeline.features.sensor_absence import extract_sensor_features
from pipeline.features.vae_decoder import extract_vae_features
from pipeline.ingest import IngestError, load_rgb
from pipeline.preprocess import JPEG_QUALITY, JPEG_SUBSAMPLING
from pipeline.provenance import analyze as provenance_analyze
from pipeline.watermark import analyze as watermark_analyze

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "models/detector_bundle.joblib"
MULTICROP_MIN_SIDE = 1024  # multi-crop helps at >=1024, hurts below (overlapping crops)

CHANNEL_EXPLAIN = {
    "v2": ("Generative-decoder artifacts",
           "Measures upsampling-grid fingerprints, spectral shape, noise structure and local "
           "self-consistency that diffusion-family generators leave in the pixels."),
    "env": ("Distance from the real-photo envelope",
            "Measures how far the image's statistics sit from the envelope of genuine camera photos; "
            "frontier generators can be flagged for being unnaturally smooth/consistent."),
}


def _strength(z):
    az = abs(z)
    return "strong" if az >= 2.0 else ("moderate" if az >= 1.0 else "weak")


class Predictor:
    def __init__(self, bundle_path=DEFAULT_BUNDLE, multicrop=True):
        self.b = joblib.load(bundle_path)
        self.multicrop = multicrop

    @staticmethod
    def _q75(crop):
        buf = io.BytesIO()
        Image.fromarray(crop, "RGB").save(buf, format="JPEG", quality=JPEG_QUALITY,
                                          subsampling=JPEG_SUBSAMPLING)
        return np.asarray(Image.open(buf).convert("RGB"), np.uint8)

    @staticmethod
    def _crops(arr, multicrop):
        H, W = arr.shape[:2]
        cy, cx = H // 2, W // 2
        anchors = [(cy - 256, cx - 256)]
        if multicrop and min(H, W) >= MULTICROP_MIN_SIDE:
            anchors += [(0, 0), (0, W - 512), (H - 512, 0), (H - 512, W - 512)]
        return [arr[y:y + 512, x:x + 512] for y, x in anchors]

    def _feats(self, crop):
        a = self._q75(crop)
        d = {}
        d.update(extract_vae_features(a))
        d.update(extract_sensor_features(a))
        d.update(extract_inversion_features(a))
        d.update(extract_selfconsistency_features(a))
        return d

    def _channel_scores(self, d):
        """Per-crop channel scores from the 27 features.

        A flat or near-uniform crop makes some features non-finite. The frozen
        bundle has no imputation for them, so the standardized vector is imputed to
        0 (the training mean, since StandardScaler centers on it). This keeps a
        degenerate crop from crashing the model or sorting NaN into the anomalous
        tail; it scores as neutral and the caller is told imputation fired.
        """
        b = self.b
        v2 = np.array([[d[k] for k in b["V2F"]]], float)
        imputed = not np.isfinite(v2).all()

        def _clean(x):
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "v2": float(b["clfV"].decision_function(_clean(b["scV"].transform(v2)))[0]),
            "env": float(b["lw"].mahalanobis(_clean(b["scE"].transform(v2)))[0]),
            "_imputed": imputed,
        }

    def _z(self, channel, score):
        """Right-tail z of a channel score against the pooled real-photo reference."""
        ref = self.b["ecdf_ref"][channel]
        p_right = 1.0 - (np.searchsorted(ref, score, side="right") / (len(ref) + 1.0))
        return float(norm.ppf(np.clip(1.0 - p_right, 1e-4, 1 - 1e-4)))

    def predict(self, path_or_array) -> dict:
        if isinstance(path_or_array, np.ndarray):
            arr, prov = path_or_array, {"verdict": "neutral", "strength": "none",
                                        "detail": "Provenance not checked (raw array input).",
                                        "checked": [], "evidence": []}
        else:
            try:
                arr = load_rgb(path_or_array)
            except IngestError as e:
                return {"verdict": "ERROR", "detail": str(e), "provenance": None}
            prov = provenance_analyze(str(path_or_array))
        H, W = arr.shape[:2]
        # The visible-watermark panel works at full resolution and its own scale
        # envelope reaches below 512px on the short side (e.g. a 16:9 stamped frame),
        # so it runs BEFORE the pixel model's size gate; only the pixel score is gated.
        wm = watermark_analyze(arr)
        if min(H, W) < 512:
            out = {"verdict": "UNSUPPORTED", "provenance": prov, "watermark": wm,
                   "verdict_basis": "pixel statistics",
                   "detail": f"Image is {W}x{H}; the forensic model needs at least 512px on the "
                             "short side and never upscales (upscaling would destroy the pixel "
                             "statistics it measures). Provenance panel is still valid."}
            if wm["found"]:
                out["verdict"] = "LIKELY AI-GENERATED"
                out["verdict_basis"] = "visible watermark: " + wm["found"][0]["mark"]
            return out
        crops = self._crops(arr, self.multicrop)
        per_crop = [self._channel_scores(self._feats(c)) for c in crops]
        imputed = any(pc["_imputed"] for pc in per_crop)
        # The median over crops already absorbs a minority of degenerate crops (say a
        # landscape with one plain-sky corner), so those need no special handling. But
        # when the majority of crops are degenerate the median is unreliable: a flat
        # image's real features (no sensor noise, very smooth) look anomalous versus a
        # real photo, i.e. AI-like, for reasons that are not generation. In that case
        # the pixel verdict is capped to INCONCLUSIVE. A watermark or provenance hit can
        # still escalate it below.
        degenerate = sum(pc["_imputed"] for pc in per_crop) * 2 > len(per_crop)
        zmed = {k: float(np.median([self._z(k, pc[k]) for pc in per_crop]))
                for k in ("v2", "env")}
        b = self.b
        s = zmed["v2"]
        null = b["real_null_scores"]
        real_pctile = float(np.searchsorted(null, s) / len(null) * 100.0)
        t_mid = float(np.quantile(null, 0.80))
        verdict = ("LIKELY AI-GENERATED" if s >= b["t_hi"]
                   else "LEANING AI-GENERATED" if s >= t_mid
                   else "LIKELY REAL" if s <= b["t_lo"] else "INCONCLUSIVE")
        basis = "pixel statistics"
        if degenerate:
            # pixel measurement unreliable in both directions -> make no pixel claim
            verdict, basis = "INCONCLUSIVE", "degenerate input (pixel model abstained)"
        if wm["found"]:
            # a generator's visible stamp at its documented position is decisive,
            # independently of the pixel score; thresholds are calibrated so that
            # no clean corner in the reference corpora ever fires
            verdict = "LIKELY AI-GENERATED"
            basis = "visible watermark: " + wm["found"][0]["mark"]
        panels = []
        for k, z in zmed.items():
            title, expl = CHANNEL_EXPLAIN[k]
            panels.append({"channel": k, "title": title, "z": round(z, 2),
                           "signal": _strength(z) if z > 0 else "none",
                           "explanation": expl})
        return {
            "verdict": verdict,
            "verdict_basis": basis,
            "watermark": wm,
            "input_size": f"{W}x{H}",
            "score_z": round(s, 3),
            "real_percentile": round(real_pctile, 1),
            "strength_text": (f"Pixel statistics are more anomalous than {real_pctile:.0f}% of "
                              "real photos in the reference corpora."),
            "bands": {"real_at_or_below": round(b["t_lo"], 3),
                      "leaning_at_or_above": round(t_mid, 3),
                      "ai_at_or_above": round(b["t_hi"], 3)},
            "n_crops": len(crops), "panels": panels, "provenance": prov,
            "degenerate_input": imputed,
            "degenerate_note": (
                "Part of this image is flat/near-uniform (e.g. a plain background, sky, "
                "or solid fill), so some pixel-statistic features were undefined and "
                "imputed to their neutral value. The pixel verdict is correspondingly "
                "less certain." if imputed else None),
            "limitations": (
                "Trained on specific generator families; strongest on FLUX/Stable-Diffusion-lineage "
                "images, weaker on the newest closed-lab generators (GPT-Image, Midjourney, Qwen, "
                "GLM). Detects generator artifacts as delivered by the generating service; "
                "social-media re-processing/laundering degrades detection. Unusual real photo "
                "classes (e.g. dashcam or heavily compressed camera frames) may be flagged as "
                "anomalous. Treat as forensic evidence, not definitive proof. Provenance metadata, "
                "when present, is reported separately and can be stripped or forged."),
        }


if __name__ == "__main__":
    import json
    print(json.dumps(Predictor(multicrop="--no-multicrop" not in sys.argv)
                     .predict([a for a in sys.argv[1:] if not a.startswith("--")][0]), indent=2))
