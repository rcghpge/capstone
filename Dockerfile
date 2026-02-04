FROM quay.io/jupyter/base-notebook:python-3.13
LABEL org.opencontainers.image.licensing="MIT AND BSD-3-Clause"
LABEL org.opencontainers.image.source="https://github.com/rcghpge/capstone"

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} \
    HOME=/home/${NB_USER} \
    NB_UID=${NB_UID}

USER root
COPY --from=ghcr.io/astral-sh/uv:latest /uv /opt/uv/bin/
ENV PATH="/opt/uv/bin:${PATH}"
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg openssl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends git tree nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    node --version | grep -q '^v20' && npm --version

RUN uv pip install --system --no-cache notebook jupyterlab kagglehub jupyterhub
RUN jupyter lab build --dev-build=False --minimize=False

WORKDIR ${HOME}
COPY --chown=${NB_UID}:0 . ${HOME}
RUN rm -rf ${HOME}/work
RUN find ${HOME} -name "*.ipynb" -exec jupyter trust {} \;
RUN uv pip install --system --no-cache -e .[dev] || uv pip install --system --no-cache -r requirements.txt
RUN find ${HOME} -name "*.egg-info" -type d -exec rm -rf {} + && \
    find ${HOME} -name "*.log*" -delete && \
    find ${HOME} -name "__pycache__" -type d -exec rm -rf {} + && \
    find ${HOME} -name "*.pyc" -delete && \
    rm -rf /root/.cache/pip /tmp/*
RUN python -m ipykernel install --sys-prefix --name python3
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache && \
    uv cache clean && rm -rf /root/.cache/uv /tmp/*
RUN fix-permissions ${HOME} /opt/conda /opt/uv || true

USER ${NB_USER}
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''", "--NotebookApp.allow_origin='*'"]
