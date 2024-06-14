import pytest
from PIL import Image
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from app.detection import Detection
from app.history import HistorySaveRequest, save_history
from app.main import app


@pytest.fixture
def api_client():
    return TestClient(app)


def get_file_name(file_path: str):
    return file_path.split('/')[-1]


def test_detect_image(api_client: TestClient, httpx_mock: HTTPXMock):
    given_file_path = 'resources/bus.jpg'
    httpx_mock.add_response()

    with open(given_file_path, 'rb') as file:
        response = api_client.post(
            '/detect/',
            files={'file': (get_file_name(given_file_path), file, 'text/plain')}
        )

    assert response.status_code == 200

    data = response.json()
    assert data['code'] == 200
    assert data['message'] == 'OK'
    assert len(data['items']) > 0


def test_detect_invalid_file_format(api_client: TestClient):
    given_file_path = 'resources/empty.txt'

    with open(given_file_path, 'rb') as file:
        response = api_client.post(
            '/detect/',
            files={'file': (get_file_name(given_file_path), file, 'text/plain')}
        )

    assert response.status_code == 200

    data = response.json()
    assert data['code'] == 400
    assert data['message'] != 'OK'
    assert not data['items']


@pytest.mark.asyncio
async def test_save_detections_api(httpx_mock: HTTPXMock):
    given_image = Image.open('resources/bus.jpg')
    given_detections = [
        Detection(xmin=671.787902, ymin=395.3720703125, xmax=810.0, ymax=878.3613, confidence=0.896, name='person'),
        Detection(xmin=12.65086, ymin=223.37843, xmax=809.7070, ymax=788.51635, confidence=0.8493, name='bus')
    ]
    httpx_mock.add_response()
    req = HistorySaveRequest(image=given_image, detections=given_detections)

    await save_history(req)

    assert req.label == 'person 1 bus 1'
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]

    assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
    for key in ['datetime', 'label', 'image']:
        assert key.encode()
