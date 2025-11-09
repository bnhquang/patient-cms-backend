from __future__ import annotations

import json
import math
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf  # noqa: F401
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (
    BASE_DIR
    / ".."
    / "public"
).resolve()

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Saved model directory not found at {MODEL_DIR}")

with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as fp:
    METADATA = json.load(fp)

NUMERIC_FEATURES = METADATA["features"]["numeric"]
CATEGORICAL_FEATURES = METADATA["features"]["categorical"]

MODEL_BUNDLE = tf.saved_model.load(MODEL_DIR)
SERVING_FN = MODEL_BUNDLE.signatures["serving_default"]
SIGNATURE_INPUT_SPECS = SERVING_FN.structured_input_signature[1]
SERVING_OUTPUT_KEY = next(iter(SERVING_FN.structured_outputs))

LABEL_CLASSES = sorted(
    METADATA.get("class_weight", {}).keys(),
    key=lambda key: (len(key), key),
)
if not LABEL_CLASSES:
    LABEL_CLASSES = ["0", "1"]

try:
    POSITIVE_INDEX = LABEL_CLASSES.index("1")
except ValueError:
    # Fallback if positive class is stored as numeric 1
    POSITIVE_INDEX = 1 if len(LABEL_CLASSES) > 1 else 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)


def _parse_datetime(value: str, field_name: str) -> pd.Timestamp:
    try:
        parsed = pd.to_datetime(value, errors="raise", utc=True)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid datetime for field '{field_name}'")
    return parsed


def _to_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required field '{field_name}'")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Field '{field_name}' must be an integer-compatible value")


def _to_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"Missing required field '{field_name}'")
    value_str = str(value).strip()
    if value_str == "":
        raise ValueError(f"Field '{field_name}' cannot be empty")
    return value_str


def preprocess_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    required_numeric = [
        "age",
        "scholarship",
        "hipertension",
        "diabetes",
        "alcoholism",
        "handicap",
        "sms_received",
    ]
    required_categorical = ["gender", "neighbourhood"]

    missing_keys = [
        key
        for key in required_numeric + required_categorical + [
            "scheduled_datetime",
            "appointment_datetime",
        ]
        if key not in payload
    ]
    if missing_keys:
        raise ValueError(f"Missing required fields: {', '.join(sorted(missing_keys))}")

    scheduled_dt = _parse_datetime(payload["scheduled_datetime"], "scheduled_datetime")
    appointment_dt = _parse_datetime(payload["appointment_datetime"], "appointment_datetime")

    wait_time_days = max((appointment_dt.normalize() - scheduled_dt.normalize()).days, 0)
    scheduled_hour = scheduled_dt.hour
    appointment_month = appointment_dt.month
    is_weekend_appointment = int(appointment_dt.weekday() >= 5)
    days_between = (appointment_dt - scheduled_dt).total_seconds() / (24 * 3600)
    days_between = float(max(days_between, 0.0))
    same_day_appointment = int(math.isclose(days_between, 0.0))

    record: Dict[str, Any] = {}

    # Numeric features
    record["Age"] = _to_int(payload["age"], "age")
    record["Scholarship"] = _to_int(payload["scholarship"], "scholarship")
    record["Hipertension"] = _to_int(payload["hipertension"], "hipertension")
    record["Diabetes"] = _to_int(payload["diabetes"], "diabetes")
    record["Alcoholism"] = _to_int(payload["alcoholism"], "alcoholism")
    record["Handicap"] = _to_int(payload["handicap"], "handicap")
    record["SMS_received"] = _to_int(payload["sms_received"], "sms_received")
    record["wait_time_days"] = wait_time_days
    record["scheduled_hour"] = scheduled_hour
    record["appointment_month"] = appointment_month
    record["is_weekend_appointment"] = is_weekend_appointment
    record["same_day_appointment"] = same_day_appointment
    record["days_between"] = days_between

    # Categorical features
    record["Gender"] = _to_string(payload["gender"], "gender")
    record["Neighbourhood"] = _to_string(payload["neighbourhood"], "neighbourhood")
    record["scheduled_day_name"] = scheduled_dt.day_name()
    record["appointment_weekday"] = appointment_dt.day_name()

    data = {feature: [record[feature]] for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    return pd.DataFrame(data)


def predict_proba(feature_df: pd.DataFrame) -> float:
    inputs = {}
    for feature_name, spec in SIGNATURE_INPUT_SPECS.items():
        if feature_name not in feature_df.columns:
            raise KeyError(f"Feature '{feature_name}' missing from feature dataframe.")

        series = feature_df[feature_name]

        if spec.dtype == tf.string:
            tensor = tf.convert_to_tensor(series.astype(str).to_numpy(), dtype=tf.string)
        elif spec.dtype.is_integer:
            np_dtype = np.int64 if spec.dtype == tf.int64 else np.int32
            tensor = tf.convert_to_tensor(series.to_numpy(dtype=np_dtype), dtype=spec.dtype)
        elif spec.dtype.is_floating:
            np_dtype = np.float32 if spec.dtype == tf.float32 else np.float64
            tensor = tf.convert_to_tensor(series.to_numpy(dtype=np_dtype), dtype=spec.dtype)
        else:
            raise TypeError(f"Unsupported dtype {spec.dtype} for feature '{feature_name}'")

        inputs[feature_name] = tensor

    outputs = SERVING_FN(**inputs)
    probs = outputs[SERVING_OUTPUT_KEY].numpy()

    if probs.ndim == 1:
        prob = float(probs[0])
    else:
        positive_idx = POSITIVE_INDEX if probs.shape[1] > POSITIVE_INDEX else probs.shape[1] - 1
        prob = float(probs[0, positive_idx])

    return prob


@app.get("/")
def health() -> Any:
    return jsonify({
        "status": "ok",
        "model": METADATA.get("model_name"),
        "last_loaded": datetime.utcnow().isoformat() + "Z",
    })


@app.post("/predict")
def predict() -> Any:
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        features_df = preprocess_payload(payload)
        probability = predict_proba(features_df)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Inference failed")
        return jsonify({"error": "Inference failed"}), 500

    feature_snapshot = features_df.iloc[0].to_dict()
    app.logger.info(
        "Inference request\nPayload: %s\nEngine features: %s\nPrediction: %.4f (%s)",
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        json.dumps(feature_snapshot, ensure_ascii=False, indent=2, default=str),
        probability,
        "will_no_show" if probability >= 0.5 else "will_show",
    )

    return jsonify({
        "no_show_probability": probability,
        "will_no_show": probability >= 0.5,
        "metadata": {
            "positive_class": LABEL_CLASSES[POSITIVE_INDEX] if LABEL_CLASSES else "1",
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

