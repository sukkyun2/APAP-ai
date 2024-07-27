import pytest
from PIL import Image
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from app.history import HistorySaveRequest, save_history
from model.detect import Detection


def get_file_name(file_path: str):
    return file_path.split('/')[-1]


def test_detect_image(api_client: TestClient, httpx_mock: HTTPXMock):
    given_file_path = 'tests/resources/bus.jpg'
    httpx_mock.add_response()

    with open(given_file_path, 'rb') as file:
        response = api_client.post(
            '/api/detect-image/',
            files={'file': (get_file_name(given_file_path), file, 'text/plain')}
        )

    assert response.status_code == 200

    data = response.json()
    assert data['code'] == 200
    assert data['message'] == 'OK'
    assert not data['item']


def test_detect_invalid_file_format(api_client: TestClient):
    given_file_path = 'tests/resources/empty.txt'

    with open(given_file_path, 'rb') as file:
        response = api_client.post(
            '/api/detect-image/',
            files={'file': (get_file_name(given_file_path), file, 'text/plain')}
        )

    assert response.status_code == 200

    data = response.json()
    assert data['code'] == 400
    assert data['message'] != 'OK'
    assert not data['item']


@pytest.mark.asyncio
async def test_save_detections_api(httpx_mock: HTTPXMock):
    given_image = Image.open('tests/resources/bus.jpg')
    given_detections = [
        Detection(class_name='person', confidence=0.884, track_id=None),
        Detection(class_name='bus', confidence=0.374, track_id=None),
    ]
    httpx_mock.add_response()
    req = HistorySaveRequest(image=given_image, detections=given_detections)

    await save_history(req)

    assert req.label == 'person 1 bus 1'
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]

    assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
    for key in ['datetime', 'label', 'image']:
        assert key.encode() in request.content
