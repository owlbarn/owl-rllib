############################################################
# owl-rllib on top of the Owl image (Owl built from its main branch).
# The Python packages for owl-rllib.bridges are not included; see the README.
############################################################

ARG BASE=matrixanger/owl:main
FROM ${BASE}

ENV OWL_RLLIB_PATH=/root/owl-rllib
COPY . ${OWL_RLLIB_PATH}
WORKDIR ${OWL_RLLIB_PATH}

# Build, test (the bridge tests are skipped without Python) and install, so
# that other projects can use (libraries owl-rllib) directly.
RUN eval $(opam env) \
    && dune build \
    && OPENBLAS_NUM_THREADS=1 dune test \
    && dune install \
    && rm -rf _build

ENV OPENBLAS_NUM_THREADS=1
ENTRYPOINT ["/bin/bash"]
