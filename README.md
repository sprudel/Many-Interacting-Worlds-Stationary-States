# MIW Demo Algorithms


## System prerequisite

This project has been tested under Ubuntu 20.04. 
The following packages are needed:

* python3
* cython3
* python3-sympy
* jupyter


## Use docker to setup environment

A docker file is provided to reproduce the environment to build and execute the conde of this repository.
This assumes you have a working docker setup installed.

To build the container run the following command. This will download and install all required dependencies:

```shell
docker build --build-arg=USER=$(id -u) -t jupyter-miw .
```

To start the jupyter notebook server run from this workspace:

```shell
docker run --mount type=bind,source="$(pwd)",target=/workspace -p 8888:8888 -it --rm jupyter-miw
```
