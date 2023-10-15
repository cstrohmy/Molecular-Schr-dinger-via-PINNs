# Molecular Schrödinger via PINNs

## Table of Contents
- [Introduction](#introduction)
- [Objective](#objective)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

This project aims to solve the time-independent molecular Schrödinger equation by employing physics-informed machine learning techniques. A cornerstone of molecular physics, understanding electron distributions within molecules is essential for ascertaining their chemical properties.

## Objective

The main objective is to find eigenvalues and eigenstates of the time-independent Schrödinger operator for the Hydrogen atom. Currently, the project generates exact solutions for the Hydrogen atom as a baseline and attempts to approximate these solutions using Physics-Informed Neural Networks (PINNs).

## Installation

1. Clone the repository to your local machine:
    \```bash
    git clone https://github.com/your_username/molecular-schrodinger-solver.git
    \```
2. Navigate to the project directory:
    \```bash
    cd molecular-schrodinger-solver
    \```
3. Install dependencies (assuming you have Python and pip installed):
    \```bash
    pip install tensorflow numpy scipy
    \```

## Usage

To generate exact solutions for the Hydrogen atom, run:

\```bash
python hydrogen.py
\```

To solve the Schrödinger equation for the Hydrogen atom using PINNs, run:

\```bash
python hydrogen-pinn.py
\```

## Results

The project currently provides exact solutions for the Hydrogen atom, serving as a benchmark. Comparison of these exact solutions with those obtained from the PINN model will be conducted as the project evolves.

## License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md) for details.
