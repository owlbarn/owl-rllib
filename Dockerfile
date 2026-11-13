############################################################
# owl-rllib on top of the Owl image (Owl built from its main branch), with
# the Python packages that owl-rllib.bridges needs.
############################################################

ARG BASE=matrixanger/owl:main
FROM ${BASE}

# Python 3.11 and the bridge's packages, installed with uv into /opt/venv.
# The image's own Python is too new for VizDoom 1.2.3. This layer is rebuilt
# only when python-requirements.txt changes.
ARG UV_VERSION=0.13.8
COPY python-requirements.txt /opt/python-requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && UV_PYTHON_INSTALL_DIR=/opt/python uv venv --python 3.11 /opt/venv \
    && uv pip install --no-cache --python /opt/venv/bin/python -r /opt/python-requirements.txt
ENV OWL_RLLIB_PYTHON=/opt/venv/bin/python

ENV OWL_RLLIB_PATH=/root/owl-rllib
COPY . ${OWL_RLLIB_PATH}
WORKDIR ${OWL_RLLIB_PATH}

# Build, test (including the bridge tests) and install, so that other
# projects can use (libraries owl-rllib) directly.
RUN eval $(opam env) \
    && dune build \
    && OPENBLAS_NUM_THREADS=1 dune test \
    && dune install \
    && rm -rf _build

ENV OPENBLAS_NUM_THREADS=1
ENTRYPOINT ["/bin/bash"]
