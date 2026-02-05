[![Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=rcghpge/capstone)

[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![pip](assets/pip-version.svg)](https://pypi.org/project/pip/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-orange?style=flat&logo=streamlit&logoColor=white)](https://health-analytics-dashboard.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcghpge/capstone/blob/main/notebooks/index.ipynb)
[![Docker](https://img.shields.io/badge/Docker-rcdpge/capstone--binder-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/rcdpge/capstone-binder)
[![Binder Container Image](https://github.com/rcghpge/capstone/actions/workflows/binder.yml/badge.svg)](https://github.com/rcghpge/capstone/actions/workflows/binder.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rcghpge/capstone/HEAD?urlpath=lab)


<div align="center">
<h1 style="font-size: 2.2em; margin: 0 0 1.5em 0; line-height: 1.2;">Capstone Project | University of Texas at Arlington</h1>

<div style="width: 100%; max-width: 1000px; margin: 0 auto 1.5em auto; padding: 2em 1em;">
  <div align="center" style="border-top: 6px solid #1f6feb; border-bottom: 6px solid #1f6feb; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; padding: 3em 2em;">
    <picture>
      <source media="(max-width: 480px)" srcset="https://raw.githubusercontent.com/lugnuts-at-UTA/graphics/refs/heads/main/img/Artboard%2022color.png">
      <source media="(max-width: 768px)" srcset="https://raw.githubusercontent.com/lugnuts-at-UTA/graphics/refs/heads/main/img/Artboard%2022color.png">
      <source media="(max-width: 1200px)" srcset="https://raw.githubusercontent.com/lugnuts-at-UTA/graphics/refs/heads/main/img/Artboard%2022color.png">
      <img src="https://raw.githubusercontent.com/lugnuts-at-UTA/graphics/refs/heads/main/img/Artboard%2022color.png" 
           alt="UTA Logo" 
           style="width: 100%; max-width: 380px; height: auto; border-radius: 16px; box-shadow: 0 20px 48px rgba(0,31,235,0.18); display: block; margin: 0 auto;" />
    </picture>
  </div>
</div>

<h2 style="font-size: 1.5em; margin: 1em 0 1.5em 0; line-height: 1.4;">
<strong>Health Analytics: Machine learning utilizing key health indicators for infant mortality rate prediction.</strong>
</h2>
</div>

<div align="left" style="max-width: 1000px; margin: 3em auto; padding: 0 2em;">
<strong>References</strong><br>
Kaggle. (2017). Annual Health Survey (India AHS 2012-13). Retrieved from<br>
<a href="https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey">
https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey
</a>
</div>

---

# Project Structure
```bash
.
├── .devontainerjson
├── .github
├── assets
├── binder
├── models
├── notebooks
├── .dockerignore
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── __init__.py
├── app.py
├── pixi.lock
├── pylock.toml
├── pyproject.toml
└── requirements.txt

4 directories, 12 files
```

---

## Getting Started
Clone the GitHub repository and generate a Pixi or Python virtual environment. Install required software dependencies.
Runs in Jupyter Notebook, Jupyter Lab, and Bash command-line environments.

```bash
# Clone repository
git clone https://github.com/rcghpge/capstone.git
cd capstone

# Generate pip venv 
python -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies 
pip install -e .[dev]

# Environment Checks
bandit -r models/ # security scanning for Python models
bandit -r models/ -f json -o security-report.json # secure report summary
pip-audit --local # audit current build environment
pip check # check for broken Python dependencies
pip install --upgrade <package> # upgrade outdated packages in build environment
pip install --upgrade -e .[dev] # upgrade build environment
python -m pip lock -e . # lock packages and dependencies in current build environment 

# Jupyterlab
jupyter lab notebooks/
jupyter lab/models/ 

# Builds with Pixi
pixi install
pixi shell
pixi info
```

## Binder
Binder provides a Jupyter development environment. This project provides a Linux-based Binder container that runs in a Jupyter development environment. Launch a Jupyter environment via Binder above or link provided below. More information about The Binder Project, a project within Project Jupyter, [here](https://jupyter.org/binder)

[Capstone binder](https://mybinder.org/v2/gh/rcghpge/capstone/HEAD?urlpath=lab) - Portable Linux-based Jupyter environment

## Docker Desktop on Microsoft Windows
[dockerdocs](https://docs.docker.com/desktop/setup/install/windows-install/) - Docker Desktop technical documentation. Full Binder image containerization for `capstone-binder` builds and JupyterLab runtime. 

## WSL on Microsoft Windows
[WSL](https://learn.microsoft.com/en-us/windows/wsl/install) - WSL technical documentation. Linux environments for containerization development on Microsoft Windows.

## Streamlit
Streamlit provides an interactive web dashboard for the KNN model simulation, exploration, and predictions. Launch the Streamlit app via the button above or link provided below. The Streamlit dashboard utilizes synthetic data for simulation and educational purposes. More information about Streamlit, an open-source Python framework for data apps, [here](https://streamlit.io)

[Streamlit dashboard](https://health-analytics-dashboard.streamlit.app/) - Streamlit interactive dashboard

---

License: MIT

---
