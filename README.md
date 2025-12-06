<div align="center">

<h1 style="font-size: clamp(28px, 5vw, 48px); font-weight: 700; color: #1f6feb; margin: 0 0 10px 0; letter-spacing: -0.02em;">
  Capstone Project 1
</h1>

<h2 style="font-size: clamp(18px, 4vw, 24px); color: #6a737d; margin: 0 0 40px 0; font-weight: 600; letter-spacing: -0.01em;">
  Division of Data Science | The University of Texas at Arlington
</h2>

<table style="width: 100%; max-width: 300px; margin: 0 auto 30px auto; border-collapse: collapse;">
  <tr>
    <td style="padding: 20px; text-align: center;">
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

<div style="max-width: 800px; margin: 0 auto 40px auto; text-align: left !important; padding-left: 20px;">

<p style="font-size: 16px; line-height: 1.6; color: #24292f; margin: 0; text-align: left !important;">
  <strong>References</strong>
</p>
<p style="font-size: 16px; line-height: 1.6; color: #24292f; margin: 0; text-align: left !important;">
  Health Analytics. India. Annual Health Survey (AHS)
</p>
<p style="font-size: 16px; line-height: 1.6; color: #24292f; margin: 0; text-align: left !important;">
  <strong>Kaggle:</strong> <a href="https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey">https://www.kaggle.com/datasets/rajanand/key-indicators-of-annual-health-survey</a>
</p>
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
├── requirements.txt
└── pyproject.yml

4 directories, 8 files
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
bandit -r models/
bandit -r models/ -f json -o security-report.jon # secure report summary
pip-audit -r requirements.txt
pip check
pytest --cov=models/ --cov-report=term-missing

# Run Python models and Launch Jupyter for EDA 
jupyter lab notebooks/ # run in a web browser environment
jupyter lab notebooks/ --no-browser # intiliaze Jupyter server with no web browser
jupyter lab/models/
jupyter lab/models/ --no-browser
```

---

License: MIT

---
