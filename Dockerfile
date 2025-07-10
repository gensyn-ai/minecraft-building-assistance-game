FROM docker.io/library/python:3.10

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY . /app

RUN /root/.local/bin/poetry build --format=wheel
RUN /root/.local/bin/poetry run auditwheel repair dist/*.whl
RUN --mount=type=secret,id=pypi_token \
    POETRY_PYPI_TOKEN_PYPI=$(cat /run/secrets/pypi_token) /root/.local/bin/poetry publish
