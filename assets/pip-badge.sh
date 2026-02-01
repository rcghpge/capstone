#!/bin/bash
set -euo pipefail

pip_version=$(pip --version | awk '{print $2}' | cut -d'(' -f1 | xargs)

python -c "
import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
from pybadges import badge
svg = badge(left_text='pip', right_text='$pip_version', left_color='grey', right_color='green')
with open('assets/pip-version.svg', 'w') as f: f.write(svg)
"
