FROM docker.io/library/python:3.10

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY . /app

RUN /root/.local/bin/poetry build --format=wheel
RUN --mount=type=secret,id=pypi_token,env=POETRY_PYPI_TOKEN_PYPI /root/.local/bin/poetry publish
