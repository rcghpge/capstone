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
    apt-get install -y --no-install-recommends curl software-properties-common && \
    curl -fsSL https://deb.nodesource.com/setup_24.x | bash - && \
    add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get upgrade -y && \  
    apt-get install --no-install-recommends git tree nodejs -y && \
    npm install -g npm@latest tar@latest validator@latest qs@latest && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /root/.npm

RUN python -m pip install --upgrade "pip>=26.0.1,<27" \
    --root-user-action=ignore \
    --disable-pip-version-check
RUN uv pip install --system --no-cache --upgrade \
    notebook jupyterlab kagglehub jupyterhub "cryptography>=46.0.5,<47"
RUN jupyter lab clean --all && \
    jupyter lab build --dev-build=True --minimize=True 

WORKDIR ${HOME}
COPY --chown=${NB_UID}:0 . ${HOME}
RUN mv ${HOME}/work ${HOME}/workspace
RUN find ${HOME} -name "*.ipynb" -exec jupyter trust {} \;

RUN uv pip install --system --no-cache -e .[dev] || uv pip install --system --no-cache -r requirements.txt
RUN find ${HOME} -name "*.egg-info" -type d -exec rm -rf {} + && \
    find ${HOME} -name "*.log*" -delete && \
    find ${HOME} -name "__pycache__" -type d -exec rm -rf {} + && \
    find ${HOME} -name "*.pyc" -delete && \
    rm -rf /root/.cache/pip /tmp/*

RUN python -m ipykernel install --sys-prefix --name python3
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache && \
    uv cache clean && rm -rf /root/.cache/uv /tmp/*
RUN fix-permissions ${HOME} "${HOME}/.local"

USER ${NB_USER}
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''", "--NotebookApp.allow_origin='*'"]
