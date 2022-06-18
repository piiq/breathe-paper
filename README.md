# BGU Respiratory Pathologies Discovery Notebooks

This repository accompanies the "Application of deep machine learning using respiratory
sound database for bronchial asthma diagnostics" paper.

The repository contains a set of python notebooks that show how to enable processing and
analysis of audio files with these tooling. The tooling utilizes the ecosystem of Python
research libraries.

## Contents

```text
├── breathe         <- Helpers and utilities library
├── data            <- Sample data provided with the repo
├── models          <- The folder where the model checkpoint files should go
│
├── README.md       <- This file
├── poetry.lock     <- Locked dependencies to reproduce environment
└── pyproject.toml  <- Project dependencies specification
```

### Notebooks

- [Database_Statistics.ipynb](Database_Statistics.ipynb). Descriptive statistics of the samples in the database
- [Audio_Preprocessing.ipynb](Audio_Preprocessing.ipynb). Illustration of the audio file pre-processing workflow
- [Inference.ipynb](Inference.ipynb). Example of running inference on an audio file

## Installation

1. Clone this repository
2. Install and activate a python environment. The reference environment uses python 3.9 installed with [`miniconda`](https://docs.conda.io/en/latest/miniconda.html).

   ```bash
   conda create -n "breathe" python=3.9 && conda activate breathe
   ```

3. Install poetry (dependency management tool).

   ```bash
   pip install poetry
   ```

4. Install dependencies with poetry.

   ```bash
   poetry install
   ```

5. Run notebooks in Jupyter Lab.

   ```bash
   jupyter lab
   ```

## Usage

- Run the notebooks in Jupyter Lab to see examples of how the tools work
- Explore the `breathe` folder to see the source code
