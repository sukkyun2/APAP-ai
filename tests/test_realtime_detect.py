import pytest
from starlette.testclient import TestClient

from app.connection_manager import PublisherAlreadyExistsError


@pytest.mark.asyncio
async def test_publisher_cannot_connect_if_already_exists(api_client: TestClient):
    with api_client.websocket_connect("/ws/publisher") as publisher:
        with pytest.raises(PublisherAlreadyExistsError):
            with api_client.websocket_connect("/ws/publisher") as another_publisher:
                pass
