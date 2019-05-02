FROM continuumio/miniconda3

# Setup a language agnostic locale
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Copy shennong sources tree
WORKDIR /shennong

# setup shennong environment
COPY environment.yml /shennong/environment.yml
RUN conda env create -n shennong -f environment.yml && \
        rm -rf /opt/conda/pkgs/*

# install/test shennong
COPY . /shennong/
RUN /bin/bash -c "source activate shennong && \
        python setup.py install && \
        python setup.py test"

# activate the shennong environment
ENV PATH=/opt/conda/envs/shennong/bin:$PATH
