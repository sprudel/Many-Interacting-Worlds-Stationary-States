# MIW Demo Algorithms For Stationary States

This repository contains demo algorithms to calculate quantum stationary states
based on the Many Interacting Worlds (MIW) approach.

This approach uses interacting worlds as a more efficient way of sampling the wave function inspired by Bohmian trajectories.
For a more detailed description of the approach see [Finding Stationary States by Interacting Quantum Worlds](https://www.mathematik.uni-muenchen.de/~deckert/publications/MSc_Herrmann_Hannes.pdf).

## Getting started

### Option A) Use docker to setup environment

This is the easiest way to setup all system requirements, as it only assumes a working docker setup. For installing docker, see the official documentation at [https://docs.docker.com/](https://docs.docker.com/).

A docker file is provided to reproduce the environment to build and execute the code of this repository.
(This should work on all x64 platforms. Most likely also on Apple ARM platforms, although this has not been tested.)

To build the container run the following command at the root of this repository. This will download and install all required dependencies:

```shell
docker build --build-arg=USER=$(id -u) -t jupyter-miw .
```

To start the jupyter notebook server run from this workspace:

```shell
docker run --mount type=bind,source="$(pwd)",target=/workspace -p 8888:8888 -it --rm jupyter-miw
```

After starting the jupyter service open the browser at the referred address and you can browse and modify the related notebooks.

### Option B) Install python environment manually

Manually setup your python environment. This project requires a standard python3 science environment including the following python dependencies: scipy, numpy, matplotlib, sympy, jupyter, jupter-notebook.

In addition you need a working setup of cython3, and a C compiler, in order to compile some optimized python code.

Be aware that most of this code has been built in 2016, so the code may need adjustments to run on the most recent python packages.
