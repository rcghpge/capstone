<div align="center">

<h2 style="font-size: clamp(20px, 4.5vw, 28px) !important; color: #6a737d; margin: 0 0 20px 0 !important; font-weight: 600 !important;">
  Capstone Project 1 | Division of Data Science | University of Texas at Arlington
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
Kaggle. Health Analytics. India. Annual Health Survey (AHS)<br>
<a href="https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey">https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey</a>
</div>
</div>

---

# Capstone Project 1 Structure
```bash
.
├── assets
├── models
├── notebooks
├── submissions
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── __init__.py
├── pylock.toml
├── pyproject.toml
├── requirements.in
└── requirements.txt

4 directories, 10 files
```

---

## Getting Started
Clone the GitHub repository and generate a Python virtual environment. Install required software dependencies.
Runs in Jupyter Notebook, Jupyter Lab, and Bash command-line environments.

```bash
# Clone repository
git clone https://github.com/rcghpge/capstoneproject.git
cd capstoneproject

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
pip-audit -r requirements.txt # audit build environment
pip check # check for broken Python dependencies
pytest --cov=models/ --cov-report=term-missing
pip list --outdated # check for outdated Python packages
pip install --upgrade <package> # upgrade outdated packages in build environment
pip install --upgrade -e . # upgrade build environment
pip freeze > requirements.txt # set requirements for current build environment
python -m venv --upgrade ~/capstoneproject # upgrade build environment with Python
python -m pip lock -e . # lock packages and dependencies in current build environment 

# Run Python models and Launch Jupyter for EDA 
jupyter lab notebooks/ # launch Jupyter Notebook in a web browser environment
jupyter lab notebooks/ --no-browser # intiliaze Jupyter server with no web browser
jupyter lab/models/ 
jupyter lab/models/ --no-browser
```

---

License: MIT

---
