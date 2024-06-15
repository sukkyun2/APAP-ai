FROM python:3.9.10-slim

ENV PYTHONUNBUFFERED 1 \
    PIP_ROOT_USER_ACTION=ignore \
    POETRY_VIRTUALENVS_CREATE=false

EXPOSE 8000
WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN pip install poetry==1.8
RUN poetry install --no-cache

COPY . ./

CMD poetry run uvicorn --host=0.0.0.0 app.main:app