# Dockerfile build for Binder
FROM quay.io/jupyter/base-notebook:python-3.12
USER root

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/
ENV PATH="/uv/bin:${PATH}"

COPY pyproject.toml requirements.txt* ./
RUN uv pip install --system --no-cache -e . || \
    uv pip install --system --no-cache -r requirements.txt
COPY . .

RUN python -m ipykernel install --sys-prefix --name python3
RUN fix-permissions $CONDA_DIR /home/$NB_USER
USER $NB_USER
