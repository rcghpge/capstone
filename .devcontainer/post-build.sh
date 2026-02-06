#!/bin/bash
set -e

/opt/conda/bin/python -m pip install --upgrade ipykernel
/opt/conda/bin/python -m ipykernel install --sys-prefix --name capstone-python3.13 --display-name "Capstone Python 3.13"
/opt/conda/bin/python -m pip install -e .[dev]

jupyter lab --generate-config --allow-root
echo 'c.ServerApp.ip = "0.0.0.0"' >> ~/.jupyter/jupyter_lab_config.py
echo 'c.ServerApp.token = ""' >> ~/.jupyter/jupyter_lab_config.py
