import pytest
import requests
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def api_client():
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_image():
    image_url = "https://ultralytics.com/images/bus.jpg"
    save_path = "bus.jpg"

    response = requests.get(image_url)
    response.raise_for_status()
    with open(save_path, 'wb') as file:
        file.write(response.content)

    yield


def test_detect_image(api_client: TestClient):
    given_file_path, given_file_name = 'bus.jpg', 'bus.jpg'

    with open(given_file_path, 'rb') as file:
        response = api_client.post(
            '/detect/',
            files={'file': (given_file_name, file, 'text/plain')}
        )

    assert response.status_code == 200

    data = response.json()
    assert data['code'] == 200
    assert data['message'] == 'OK'
    assert len(data['items']) > 0


def test_detect_invalid_file_format(api_client: TestClient):
    def make_empty_text_file():
        with open(given_file_path, 'w', encoding='utf-8'):
            pass

    given_file_path, given_file_name = 'other.txt', 'other.txt'
    make_empty_text_file()

    with open(given_file_path, 'rb') as file:
        response = api_client.post(
            '/detect/',
            files={'file': (given_file_name, file, 'text/plain')}
        )

    assert response.status_code == 200

    data = response.json()
    assert data['code'] == 400
    assert data['message'] != 'OK'
    assert not data['items']
