import pytest
from starlette.testclient import TestClient

from app.main import app


@pytest.fixture
def api_client():
    return TestClient(app)
