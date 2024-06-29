# APAP-AI
[![Deploy with Docker](https://github.com/sukkyun2/APAP-ai/actions/workflows/deploy.yml/badge.svg)](https://github.com/sukkyun2/APAP-ai/actions/workflows/deploy.yml)

This is a repository for an object detection inference API using `YOLOv5` and `FastAPI`

This repository is used for 스마트해상물류 ICT project.


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

| Variable             | Description                        |
|----------------------|------------------------------------|
| GHCR_TOKEN           | GitHub Container Registry Token    |
| REMOTE_IP            | Remote server IP                   |
| REMOTE_USER          | Remote server user name            |
| REMOTE_PRIVATE_KEY   | Remote server private key          |
| REMOTE_SSH_PORT      | 22                                 |


