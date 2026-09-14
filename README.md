# ECE 592 — Fall 2026

This repository contains notebooks, code, and supporting material for ECE 592. Most of the notebooks are to combine short theoretical discussions with executable examples, visualizations, and experiments that can be modified during or after class.

The contents of this repository will continue to change as the course progresses.

## Running the notebooks

The simplest option is Google Colab. A local Python installation is also supported for anyone who wants to modify the code more extensively.

### Google Colab

Notebooks that support Colab include an **Open in Google Colab** badge near the top.

The quantum data-encoding notebook can be opened here:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegacYFTw/ece592-fall-2026/blob/main/encoding_schemes.ipynb)


### Local installation with pip

Clone the repository and enter the project directory:

```bash
git clone https://github.com/LegacYFTw/ece592-fall-2026.git
cd ece592-fall-2026
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The notebooks can then be opened with JupyterLab, VS Code, PyCharm, or another Jupyter-compatible editor.

If JupyterLab is not already installed, run:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

### Local installation with `uv`

If `uv` is already installed, the environment can be reproduced directly from the project configuration:

```bash
uv sync
```

The notebooks can then be opened using the Python environment created by `uv`.

Using `uv` is optional. Students do not need it to run the course material.

## Dependency files

The repository contains several environment files for different purposes:

* `pyproject.toml` lists the project’s direct dependencies.
* `uv.lock` records the complete environment used during development.
* `requirements.txt` provides a pip-compatible version of the project environment.
* `requirements-colab.txt` contains the smaller collection of packages required inside Google Colab.

The dependency versions are pinned because scientific Python and Qiskit interfaces can change between releases. Using the supplied files avoids subtle differences between student environments.


## Data

The notebooks use publicly available datasets whenever possible. Dataset downloads are performed from within the notebooks, so data files generally do not need to be committed to the repository.

Some datasets are obtained through Kaggle using `kagglehub`. Public datasets should download automatically. If Kaggle changes its access requirements, the corresponding notebook may ask for authentication.

## Computational expectations

Most examples use exact simulation so that the behavior of the circuits can be studied without hardware noise.

Small circuit demonstrations should run quickly. Quantum-kernel matrices and repeated VQC training can take several minutes, particularly in Colab. Progress bars are included for the longer calculations.

No IBM Quantum account or API token is required unless a notebook explicitly introduces hardware execution.

