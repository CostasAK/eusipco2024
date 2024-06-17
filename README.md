# Transmit and Receive Sensor Selection Using the Multiplicity in the Virtual Array

[<img src="https://media.licdn.com/dms/image/D4D0BAQGI3u-J_KWMoA/company-logo_200_200/0/1706800320569/eusipco_logo?e=1726704000&v=beta&t=7LHnNirAwhMmAZVz0c3QYYvG4WW5HC6cnymW7vOyN0k" align="right" max-width="200px" alt="EUSIPCO 2024 Logo"/>](https://eusipcolyon.sciencesconf.org)
By Ids van der Werf, Costas A. Kokke, Richard Heusdens, Richard C. Hendriks, Geert Leus, and Mario Coutino.

Code accompanying our submission to the 32nd European Signal Processing Conference (EUSIPCO 2024).

## Abstract

> The main focus of this paper is an active sensing application that involves selecting transmit and receive sensors to optimize the Cramér-Rao bound (CRB) on target parameters. Although the CRB is non-convex in the transmit and receive selection, we demonstrate that it is convex in the virtual array weight vector, which describes the multiplicity of the virtual array elements. Based on this finding, we propose a novel algorithm that optimizes the virtual array weight vector first and then finds a matching transceiver array. This greatly enhances the efficiency of the transmit and receive sensor selection problem.

## Viewing

The notebook can be viewed online by opening it in nbviewer or Google Colab.

[![Open in nbviewer](https://img.shields.io/static/v1?label&message=Open+in+nbviewer&color=343433&style=for-the-badge&logo=jupyter)](https://nbviewer.org/github/CostasAK/eusipco2024/blob/main/main.ipynb)
[![Open in Colab](https://img.shields.io/static/v1?label&message=Open+in+Colab&color=097ABB&style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/CostasAK/eusipco2024/blob/main/main.ipynb)

## Usage

Tested using Pipenv and Jupyter in Visual Studio Code on Ubuntu 20.04.

1. `git clone` this repository and `cd` into the directory.
2. (optional) `export PIPENV_VENV_IN_PROJECT=1` to install Pipenv virtual environments into the current project folder.
3. `pipenv install`.
4. Open this folder in Visual Studio Code.
5. Install the workspace recommended extension.
6. Open `main.ipynb`.

Alternatively, you can try and run a Jupyter server manually, or use Google Colab.
