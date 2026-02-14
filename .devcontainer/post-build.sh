#!/bin/bash
set -e

echo 'export PATH="/home/jovyan/.local/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"' > ~/.bashrc
source ~/.bashrc

/opt/conda/bin/python -m pip install --upgrade --user ipykernel
/opt/conda/bin/python -m ipykernel install --sys-prefix --name capstone-python3.13 --display-name "Capstone Python 3.13"
/opt/conda/bin/python -m pip install --user -e .[dev]
rm -rf *.egg-info

jupyter lab --generate-config --allow-root || true
echo 'c.ServerApp.ip = "0.0.0.0"' >> ~/.jupyter/jupyter_lab_config.py
echo 'c.ServerApp.token = ""' >> ~/.jupyter/jupyter_lab_config.py

conda init bash --user
echo 'conda activate base' >> ~/.bashrc

echo "✅ Setup done!"
