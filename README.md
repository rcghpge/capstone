[![Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=rcghpge/capstone)

[![Binder Container Image](https://github.com/rcghpge/capstone/actions/workflows/binder.yml/badge.svg)](https://github.com/rcghpge/capstone/actions/workflows/binder.yml)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rcghpge/capstone/main?urlpath=lab)

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

<p style="font-size: clamp(25px, 7vw, 33px); line-height: 1.6; color: #24292f; max-width: 800px; margin: 0 auto 20px auto;">
 <strong>Machine learning utilizing key health indicators for infant mortality rate prediction.</strong><br>
</p>
</div>
<!--- <div style="max-width: 800px; margin: 0 auto 40px auto; text-align: left !important; padding-left: 20px; line-height: 1.4;">
<strong>References</strong><br>
Kaggle. Health Analytics. India. Annual Health Survey (AHS)<br>
<a href="https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey">https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey</a>
</div>
</div> --->

---

# Project Structure
```bash
.
├── assets
├── models
├── notebooks
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

3 directories, 11 files
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
Binder provides a development environment but takes a bit to launch. Launch a Jupyter environment via Binder above or link provided below. More information about The Binder Project, a project within Project Jupyter [here](https://jupyter.org/binder)

[Binder](https://mybinder.org/v2/gh/rcghpge/capstone/main?urlpath=lab) - Not fully tested. Still in development for stable workflow environments

---

License: MIT

---
