FROM docker.io/library/python:3.11

RUN curl -sSL https://install.python-poetry.org | python3.10 -

WORKDIR /app

COPY . /app

RUN /root/.local/bin/poetry run cibuildwheel
RUN --mount=type=secret,id=pypi_token \
    POETRY_PYPI_TOKEN_PYPI=$(cat /run/secrets/pypi_token) /root/.local/bin/poetry publish
