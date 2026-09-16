# tests/test_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.anyio
async def test_predict_success():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "features": [3.5, 1.2, 4.9]
    })
    assert resp.status_code == 200
    assert {"predictions": [7.0, 2.4, 9.8]} == resp.json()

@pytest.mark.anyio
async def test_predict_unprocessable_entity():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "feature1": 3.5,
        "feature2": 1.2,
        "feature3": 4.9
    })
    assert resp.status_code == 422


# Un test qui valide une prédiction correcte
@pytest.mark.anyio
async def test_predict_correct_values():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
            "features": [1.0, 2.0, 3.0]
        })

    assert resp.status_code == 200
    assert {"predictions": [2.0, 4.0, 6.0]} == resp.json()

# Un test qui valide une prédiction incorrecte
@pytest.mark.anyio
async def test_predict_incorrect():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
            "features": [1.0, 2.0, 3.0]
        })

    assert resp.status_code == 200

    # Résultat volontairement faux
    expected = {"predictions": [2.0, 4.0, 7.0]}

    assert resp.json() != expected

# Test avec le champ "features" manquant
@pytest.mark.anyio
async def test_predict_missing_features():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
            "values": [3.5, 1.2, 4.9]
        })

    assert resp.status_code == 422