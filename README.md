[![Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=rcghpge/capstone)

[![Docker](https://img.shields.io/badge/Docker-rcdpge/capstone--binder-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/rcdpge/capstone-binder)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcghpge/capstone/blob/main/notebooks/index.ipynb)
[![Binder Container Image](https://github.com/rcghpge/capstone/actions/workflows/binder.yml/badge.svg)](https://github.com/rcghpge/capstone/actions/workflows/binder.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rcghpge/capstone/HEAD?urlpath=lab)

<div align="center">

<h2 style="font-size: clamp(20px, 4.5vw, 28px) !important; color: #6a737d; margin: 0 0 20px 0 !important; font-weight: 600 !important;">
  Capstone Project | Division of Data Science | University of Texas at Arlington
</h2>

<table style="width: 100%; margin: 40px auto 30px auto; border-collapse: collapse;">
  <tr>
    <td style="padding: 20px 0; text-align: center; border-top: 3px solid #1f6feb; border-bottom: 3px solid #1f6feb;">
      <img src="assets/UTA Celebrating 130 Years logo white circle.png" 
           alt="UTA Logo" 
           style="width: 100%; height: auto; max-width: 200px; border-radius: 50%; box-shadow: 0 8px 24px rgba(0,0,0,0.15);" />
    </td>
  </tr>
</table>

<p style="font-size: clamp(16px, 3.5vw, 20px); line-height: 1.6; color: #24292f; max-width: 800px; margin: 0 auto 20px auto;">
  Machine learning utilizing key health indicators for infant mortality rate prediction.
</p>
</div>
<div style="max-width: 800px; margin: 0 auto 40px auto; text-align: left !important; padding-left: 20px; line-height: 1.4;">
<strong>References</strong><br>
Kaggle. (2017). Annual Health Survey (India AHS 2012-13) Retrieved September 12, 2025, from<br>
<a href="https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey">https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey</a>
</div>
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
Clone the GitHub repository and generate a Python virtual environment. Install required software dependencies.
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
python -c "from models import *; print('✅ Model import dependencies OK')"
bandit -r . # scan current build environment
bandit -r models/ # scan Python models
bandit -r models/ -f json -o security-report.json # secure report summary
pip-audit --local # audit current build environment
pip check # check for broken Python dependencies
pytest --cov=models/ --cov-report=term-missing
pip list --outdated # check for outdated Python packages
pip install --upgrade <package> # upgrade outdated packages in build environment
pip install --upgrade -e . # upgrade build environment
pip freeze > requirements.txt # set requirements for current build environment
python -m venv --upgrade ~/capstone # upgrade build environment with Python
python -m pip lock -e . # lock packages and dependencies in current build environment 

# Run Python models and Launch Jupyter for EDA 
jupyter lab notebooks/ # launch Jupyter Notebook in a web browser environment
jupyter lab notebooks/ --no-browser # intiliaze Jupyter server with no web browser
jupyter lab/models/ 
jupyter lab/models/ --no-browser

# Builds with Pixi
pixi shell
pixi info
```

## Binder
Binder provides a Jupyter development environment. Launch a Jupyter environment via Binder above or link provided below. More information about The Binder Project, a project within Project Jupyter [here](https://jupyter.org/binder)

[Capstone Binder](https://mybinder.org/v2/gh/rcghpge/capstone/HEAD?urlpath=lab) - Still wip

---

License: MIT

---
