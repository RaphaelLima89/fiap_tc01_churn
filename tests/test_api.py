"""API test: POST /predict com payload valido retorna resposta correta."""

from __future__ import annotations


def test_predict_endpoint_with_valid_payload(client, valid_payload) -> None:
    """End-to-end: TestClient -> /predict -> 200 com formato esperado."""
    response = client.post("/predict", json=valid_payload)

    # Status e content-type
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()

    # Formato da resposta (PredictionResponse)
    assert set(body.keys()) == {
        "request_id",
        "churn_probability",
        "churn_prediction",
        "threshold",
        "model_version",
        "latency_ms",
    }

    # Ranges
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["threshold"] == 0.08  # API_DECISION_THRESHOLD
    assert body["latency_ms"] >= 0.0
    assert body["model_version"]  # nao vazio
    assert body["request_id"]  # nao vazio

    # Headers customizados dos middlewares
    assert "x-request-id" in response.headers
    assert "x-response-time-ms" in response.headers
    assert response.headers["x-request-id"] == body["request_id"]

    # Coerencia threshold/predicao
    if body["churn_probability"] >= body["threshold"]:
        assert body["churn_prediction"] == 1
    else:
        assert body["churn_prediction"] == 0
