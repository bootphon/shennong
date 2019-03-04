FROM pykaldi/pykaldi:latest

# Setup a language agnostic locale
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install shennong dependencies
RUN pip --no-cache-dir install \
        cython \
        h5py \
        pytest \
        pytest-cov \
        pytest-runner \
        git+https://github.com/bootphon/h5features.git

# Install shennong
WORKDIR /shennong
COPY . .
RUN python setup.py install
RUN python setup.py test
