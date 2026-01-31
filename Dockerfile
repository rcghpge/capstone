FROM quay.io/jupyter/base-notebook:python-3.13
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} \
    HOME=/home/${NB_USER} \
    NB_UID=${NB_UID}

USER root
COPY --from=ghcr.io/astral-sh/uv:latest /uv /opt/uv/bin/
ENV PATH="/opt/uv/bin:${PATH}"
RUN uv pip install --system --no-cache notebook jupyterlab jupyterhub
RUN jupyter lab build --dev-build=False --minimize=False || true
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates openssl nodejs npm && rm -rf /var/lib/apt/lists/*

WORKDIR ${HOME}
COPY --chown=${NB_UID}:0 . ${HOME}

RUN uv pip install --system --no-cache -e .[dev] || uv pip install --system --no-cache -r requirements.txt
RUN python -m ipykernel install --sys-prefix --name python3
RUN fix-permissions ${HOME} /opt/conda /opt/uv || true && rm -rf /tmp/*

USER ${NB_USER}
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''", "--NotebookApp.allow_origin='*'"]
