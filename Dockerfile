FROM ubuntu:20.04
ARG USER
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 jupyter jupyter-notebook  \
    python3-sympy python3-numpy python3-matplotlib python3-scipy \
    cython3 build-essential python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /workspace
RUN adduser jupyter-user --uid ${USER}
RUN chown -R jupyter-user /workspace

USER jupyter-user

WORKDIR /workspace

ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0"]
