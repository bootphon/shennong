# Use an official Ubuntu as a parent image
FROM ubuntu:18.04

# Set the working directory to /shennong
WORKDIR /shennong

# Install dependencies from the ubuntu repositories
RUN apt-get update && \
    apt-get install -y \
    autoconf \
    automake \
    cmake \
    curl \
    cython3 \
    g++ \
    git \
    graphviz \
    libatlas3-base \
    libatlas-base-dev \
    libtool \
    make \
    ninja-build \
    pkg-config \
    python2.7 \
    python3 \
    python3-h5py \
    python3-numpy \
    python3-scipy \
    python3-pip \
    python3-pyparsing \
    python3-pytest \
    python3-setuptools \
    subversion \
    unzip \
    wget \
    zlib1g-dev

# Make python3 default
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# Install python packages from pip
RUN pip install --upgrade pip git+https://github.com/bootphon/h5features.git

# Clone, compile and install pykaldi
RUN git clone https://github.com/pykaldi/pykaldi.git /pykaldi

RUN cd /pykaldi/tools \
    && ./check_dependencies.sh \
    && ./install_protobuf.sh \
    && ./install_clif.sh \
    && ./install_kaldi.sh

RUN cd /pykaldi \
    && python setup.py install \
    && python setup.py test

# Install shennong
COPY . /shennong
RUN python setup.py install
RUN python setup.py test
