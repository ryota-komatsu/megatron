# megatron

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org)

## Installation
```bash
conda create -y -n py312 -c pytorch -c nvidia -c conda-forge python=3.12 transformer-engine-torch causal-conv1d
pip install --no-build-isolation -r requirements.txt
```