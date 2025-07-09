FROM docker.io/library/python:3.10

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY . /app

RUN /root/.local/bin/poetry build --format=wheel
