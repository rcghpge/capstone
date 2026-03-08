[![Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=rcghpge/capstone)

[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Conda](https://img.shields.io/badge/conda-%3E%3D26.1.0-blue?logo=anaconda&logoColor=white)](https://docs.conda.io/en/latest/)
[![pip](assets/pip-version.svg)](https://pypi.org/project/pip/)
[![uv](https://img.shields.io/badge/uv-0.10.6-blue)](https://github.com/astral-sh/uv)
[![Streamlit](https://img.shields.io/badge/Streamlit-Simulation-orange?style=flat&logo=streamlit&logoColor=white)](https://health-analytics-dashboard.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcghpge/capstone/blob/main/notebooks/index.ipynb)
[![Docker](https://img.shields.io/badge/Docker-rcdpge/capstone--binder-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/rcdpge/capstone-binder)
[![Binder Container Image](https://github.com/rcghpge/capstone/actions/workflows/binder.yml/badge.svg)](https://github.com/rcghpge/capstone/actions/workflows/binder.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rcghpge/capstone/main?urlpath=lab)

<div align="center">
<h2 style="font-size: 2.2em; margin: 0 0 1.5em 0; line-height: 1.2;">Capstone Project | University of Texas at Arlington</h2>

<div style="width: 100%; max-width: 1000px; margin: 0 auto 1.5em auto; padding: 2em 1em;">
  <div align="center" style="border-top: 6px solid #1f6feb; border-bottom: 6px solid #1f6feb; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; padding: 3em 2em;">
    <picture>
      <source media="(max-width: 480px)" srcset="https://raw.githubusercontent.com/lugnuts-at-UTA/graphics/refs/heads/main/img/Artboard%2022color.png">
      <source media="(max-width: 768px)" srcset="https://raw.githubusercontent.com/lugnuts-at-UTA/graphics/refs/heads/main/img/Artboard%2022color.png">
      <source media="(max-width: 1200px)" srcset="https://raw.githubusercontent.com/lugnuts-at-UTA/graphics/refs/heads/main/img/Artboard%2022color.png">
      <img src="https://raw.githubusercontent.com/lugnuts-at-UTA/graphics/refs/heads/main/img/Artboard%2022color.png" 
           alt="UTA Logo" 
           style="width: 100%; max-width: 380px; height: auto; border-radius: 16px; box-shadow: 0 20px 48px rgba(96, 165, 250, 0.40); display: block; margin: 0 auto;" />
    </picture>
  </div>
</div>

<h3 style="font-size: 1.5em; margin: 1em 0 1.5em 0; line-height: 1.4;">
<strong>Health Analytics: Machine learning utilizing key health indicators for infant mortality rate prediction.</strong>
</h3>
</div>

<div align="left" style="max-width: 1000px; margin: 3em auto; padding: 0 2em;">
<strong>References</strong><br>
Kaggle (2017). Annual Health Survey (India AHS 2012-13):<br>
<a href="https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey">
https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey
</a>
</div>

---

## Project Structure
```bash
.
├── .devontainer
├── .github
├── assets
├── binder
├── models
├── notebooks
├── .dockerignore
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── Dockerfile
├── LICENSE
├── README.md
├── __init__.py
├── app.py
├── pixi.lock
├── pylock.toml
├── pyproject.toml
└── requirements.txt

7 directories, 13 files
```

---

## Getting Started
Clone the GitHub repository and generate a Pixi or Python virtual environment. Install required software dependencies. 
Developed for Jupyter Notebooks, Jupyter Lab, Docker Desktop, Binder containers, and Bash CLI environments.

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
bandit -r models/ # secure Python scanning
pip-audit  # audit current build environment
pip check # check for broken Python dependencies
pip install --upgrade <package> # upgrade packages
pip install --upgrade -e .[dev] # upgrade development environment
python -m pip lock -e . # lock packages and dependencies  

# Jupyterlab
jupyter lab
jupyter lab notebooks/
jupyter lab/models/ 

# Builds with Pixi
pixi install
pixi shell
pixi info
```

## Streamlit
[Streamlit](https://streamlit.io/) - Streamlit is the fastest way to build and share data apps, letting you transform Python scripts into interactive web applications with just a few lines of code.

[Streamlit dashboard](https://health-analytics-dashboard.streamlit.app/) - Streamlit interactive simulation dashboard

## Binder
[Binder](https://jupyter.org/binder) - Binder is an open-source web service that allows you to build, share, and run interactive computational environments in the cloud directly from your Git repositories.

[Capstone binder](https://mybinder.org/v2/gh/rcghpge/capstone/main?urlpath=lab) - Binder container image for capstone project with a portable Linux-based Jupyter environment.

## Docker Desktop 
[Docker Desktop](https://docs.docker.com/desktop/) - Docker Desktop is a one-click-install application for your Mac, Linux, or Windows environment that lets you build, share, and run containerized applications and microservices. 

### Binder Containers on Docker Desktop

Run:
```bash
docker run -it -p 8888:8888 rcdpge/capstone-binder:latest
```
Access Jupyter at http://localhost:8888 (token printed in logs).


## Windows Subsystem for Linux (WSL)
[Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) - Windows Subsystem for Linx is an integrated Linux environment for cross-platform development on Microsoft Windows without virtual machines or dual-booting.


---

License: MIT

---
