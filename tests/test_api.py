import pytest
from httpx import AsyncClient

from app.main import app


@pytest.mark.anyio
async def test_predict_success() -> None:
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={"features": [3.5, 1.2, 4.9]},
        )
        
    assert resp.status_code == 200
    assert resp.json() == {"predictions": [7.0, 2.4, 9.8]}


@pytest.mark.anyio
async def test_predict_unprocessable_entity() -> None:
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={
                "feature1": 3.5,
                "feature2": 1.2,
                "feature3": 4.9,
            },
        )

    assert resp.status_code == 422


@pytest.mark.anyio
async def test_predict_correct_values() -> None:
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0]},
        )

    assert resp.status_code == 200
    assert resp.json() == {"predictions": [2.0, 4.0, 6.0]}


@pytest.mark.anyio
async def test_predict_missing_features() -> None:
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={"values": [3.5, 1.2, 4.9]},
        )

    assert resp.status_code == 422