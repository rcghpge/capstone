# Dockerfile build for Binder
FROM jupyter/base-notebook:python-3.12-slim

USER root

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
RUN python -m ipykernel install --sys-prefix --name python
RUN fix-permissions $CONDA_DIR /home/$NB_USER
USER $NB_USER
