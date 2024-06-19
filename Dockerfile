FROM python:3.9.10-slim

ENV PYTHONUNBUFFERED 1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

EXPOSE 8000
WORKDIR /app

# opencv
RUN apt-get update -y && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY poetry.lock pyproject.toml ./

RUN pip install poetry==1.8

# install dependencies and remove poetry cache
RUN poetry install && rm -rf $POETRY_CACHE_DIR

COPY . ./

CMD poetry run uvicorn --host=0.0.0.0 app.main:app