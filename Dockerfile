FROM python:3.11-slim AS builder

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# to run poetry directly as soon as it's installed
ENV PATH="$POETRY_HOME/bin:$PATH"

# install poetry
RUN pip install poetry==1.8

WORKDIR /app

# copy only pyproject.toml and poetry.lock file nothing else here
COPY poetry.lock pyproject.toml ./

# this will create the folder /app/.venv (might need adjustment depending on which poetry version you are using)
RUN poetry install --no-root --no-ansi --without test

FROM python:3.9-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install openCV related packages
RUN apt-get update -y && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# copy the venv folder from builder image
COPY --from=builder /app/.venv ./.venv
COPY . ./

EXPOSE 8000
CMD python -m uvicorn --host=0.0.0.0 app.main:app