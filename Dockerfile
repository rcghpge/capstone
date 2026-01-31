FROM quay.io/jupyter/base-notebook:python-3.13
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} HOME=/home/${NB_USER}

USER root
COPY --from=ghcr.io/astral-sh/uv:latest /uv /opt/uv/bin/
ENV PATH="/opt/uv/bin:${PATH}"
RUN uv pip install --system --no-cache-dir jupyterlab jupyterhub

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates openssl \
    && update-ca-certificates \
    && curl -s https://api.github.com/repos/jupyterhub/repo2docker-action/releases/latest || true \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${HOME}
COPY --chown=${NB_UID}:${NB_UID} . ${HOME}
RUN rm -rf ${HOME}/.git ${HOME}/__pycache__

RUN python -m pip install --upgrade pip
RUN uv pip install --system --no-cache -e . || uv pip install --system --no-cache -r requirements.txt

RUN python -m ipykernel install --sys-prefix --name python3
RUN fix-permissions ${HOME} /opt/conda /opt/uv \
    && rm -rf /tmp/*

USER ${NB_USER}
#BINDER COMPAT: Inherit base-notebook entrypoint
#ENTRYPOINT ["start-notebook.py"]
#full CMD:
#CMD ["start-notebook.py", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''"]
