# APAP-AI

[![Deploy with Docker](https://github.com/sukkyun2/APAP-ai/actions/workflows/deploy.yml/badge.svg)](https://github.com/sukkyun2/APAP-ai/actions/workflows/deploy.yml)

This is a repository for an object detection inference API using `YOLOv8` and `FastAPI`

This repository is used for 스마트해상물류 ICT project.

## How to Use

### Operations
There are three operations: `estimate_distance`, `custom_model`, and `area_intrusion`. 
Each of these operations can be used to detect different anomalies in video.

| Operation          | Description                     |
|--------------------|---------------------------------|
| estimate_distance  | GitHub Container Registry Token |
| custom_model       | Remote server IP                |
| area_intrusion     | Remote server user name         |

### PUB-SUB
In this system, a publisher (camera) sends frames to multiple subscribers (users) via WebSocket. 
Subscribers receive processed frames based on the operation specified. For example:

#### Publisher
```shell
{host}/ws/publishers/{location_name}?op={operation_name}
```
- `{host}`: APAP AI server address
- `{location_name}`: Camera location or context
- `{operation_name}`: Operation to apply (e.g., estimate_distance)

#### Subscriber
```shell
{host}/ws/subscribers/{location_name}
```
- `{host}`: APAP AI server address
- `{location_name}`: Camera location or context


   
<img src="https://github.com/user-attachments/assets/3993ee1a-af1e-46a4-aea9-951a74fb76b5" alt="image" width="800"/>

---

## How to Run

### Install

Dependencies are managed using `Poetry`.

```shell
 pip install poetry
 poetry install
```

### Run

```shell
poetry run uvicorn --host=127.0.0.1 app.main:app
```

### Usage

Environment variables are managed through dotenv.

The defined variables are referenced from the `.env` file in the root directory.

### API Endpoints

Documentation is provided through `SwaggerUI`, which is built into `FastAPI` by default.

```http request
 http://localhost:8080/docs
```

### Test

```shell
 poetry run pytest
```

### Deployment via GitHub Actions

The following variables need to be defined in the `Environment secrets`:

| Variable           | Description                     |
|--------------------|---------------------------------|
| GHCR_TOKEN         | GitHub Container Registry Token |
| REMOTE_IP          | Remote server IP                |
| REMOTE_USER        | Remote server user name         |
| REMOTE_PRIVATE_KEY | Remote server private key       |
| REMOTE_SSH_PORT    | 22                              |



