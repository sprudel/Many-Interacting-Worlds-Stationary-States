FROM ubuntu:22.04
ARG USER
RUN apt-get update
RUN apt-get install -y \
    python3 \
    cython3 \
    jupyter \
    jupyter-notebook

RUN mkdir /workspace
RUN adduser jupyter-user --uid ${USER}
RUN chown -R jupyter-user /workspace

USER jupyter-user

WORKDIR /workspace

ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0"]
