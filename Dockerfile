FROM quay.io/jupyter/base-notebook:python-3.12
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} HOME=/home/${NB_USER}

USER root
COPY --from=ghcr.io/astral-sh/uv:0.4.18 /uv /opt/uv/bin/
ENV PATH="/opt/uv/bin:${PATH}"
WORKDIR ${HOME}

COPY --chown=${NB_UID}:${NB_UID} pyproject.toml* requirements.txt* ${HOME}/
RUN uv pip install --system --no-cache -e .[dev] || uv pip install --system --no-cache -r requirements.txt

# Kernel + final copy
RUN python -m ipykernel install --sys-prefix --name python3
COPY --chown=${NB_UID}:${NB_UID} . ${HOME}
RUN fix-permissions ${HOME} /opt/uv && rm -rf /tmp/*

USER ${NB_USER}
