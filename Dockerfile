# jupyter lab dockerfile

FROM python:3.10.4-slim-bullseye

RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y gcc python3-dev

# Install kernels
RUN pip install jupyter jupyterlab

RUN pip install bash_kernel
RUN python -m bash_kernel.install

WORKDIR /notebooks

COPY requirements.txt .
RUN pip install -r requirements.txt

ARG ARG_GID
ARG ARG_UID
ENV GID=$ARG_GID
ENV UID=$ARG_UID
RUN adduser -q --uid $UID --gid $GID alien
USER alien

EXPOSE 8888
