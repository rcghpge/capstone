FROM quay.io/jupyter/base-notebook:python-3.13
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} HOME=/home/${NB_USER}

USER root
COPY --from=ghcr.io/astral-sh/uv:latest /uv /opt/uv/bin/
ENV PATH="/opt/uv/bin:${PATH}"
RUN uv pip install --system --no-cache-dir jupyterlab jupyterhub

WORKDIR ${HOME}
COPY --chown=${NB_UID}:${NB_UID} . ${HOME}
RUN rm -rf ${HOME}/.git ${HOME}/__pycache__

RUN python -m pip install --upgrade pip
RUN uv pip install --system --no-cache -e . || uv pip install --system --no-cache -r requirements.txt

RUN python -m ipykernel install --sys-prefix --name python3
RUN fix-permissions ${HOME} /opt/uv /home/${NB_USER}/.cache \
    && rm -rf /tmp/* /home/${NB_USER}/.cache/pip*

USER ${NB_USER}
