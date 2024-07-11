import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.connection_manager import PublisherAlreadyExistsError


@pytest.mark.asyncio
async def test_websocket_pub_sub(api_client: TestClient):
    with api_client.websocket_connect("/ws/publisher") as publisher:
        with api_client.websocket_connect("/ws/subscriber") as subscriber:
            # given
            dummy_data = np.zeros((100, 100, 3), dtype=np.uint8)
            _, dummy_buffer = cv2.imencode('.jpg', dummy_data)

            # when
            publisher.send_bytes(dummy_buffer.tobytes())
            processed_data = subscriber.receive_bytes()

            # then
            processed_img_np = np.frombuffer(processed_data, np.uint8)
            processed_img = cv2.imdecode(processed_img_np, cv2.IMREAD_COLOR)
            assert processed_img is not None


@pytest.mark.asyncio
async def test_publisher_cannot_connect_if_already_exists(api_client: TestClient):
    with api_client.websocket_connect("/ws/publisher") as publisher:
        with pytest.raises(PublisherAlreadyExistsError):
            with api_client.websocket_connect("/ws/publisher") as another_publisher:
                pass
