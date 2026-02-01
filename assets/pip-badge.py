#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', message='pkg_resources')
import subprocess, re
from pybadges import badge

result = subprocess.run(['pip', '--version'], capture_output=True, text=True)
match = re.search(r'pip (\S+)', result.stdout)
pip_version = match.group(1) if match else 'unknown'

svg = badge(left_text='pip', right_text=pip_version, left_color='grey', right_color='green')
with open('assets/pip-version.svg', 'w') as f:
    f.write(svg)
