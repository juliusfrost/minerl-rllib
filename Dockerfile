FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime AS dependencies

ENV POETRY_VERSION=1.1.8 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    MINERL_HEADLESS=1

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    xvfb \
    x11-xserver-utils \
    openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /minerl-rllib
COPY poetry.lock pyproject.toml /minerl-rllib/

RUN pip install poetry==$POETRY_VERSION
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

COPY tests/build_minerl.py /minerl-rllib/tests/build_minerl.py
RUN python /minerl-rllib/tests/build_minerl.py

COPY . /minerl-rllib

FROM dependencies AS train
# Installs the minerl-rllib package
RUN poetry install --no-interaction --no-ansi
#ENTRYPOINT ["python", "minerl_rllib/rllib_train.py"]
#CMD ["-f bc.yaml"]