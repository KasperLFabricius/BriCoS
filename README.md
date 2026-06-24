# BriCoS (Bridge Comparison Software)

**BriCoS** is a lightweight, browser-based Finite Element Method (FEM) tool for rapid comparison of two 2D bridge systems.

## Key Features

* **Dual System Analysis:** Model, analyze, and visualize two structural systems side by side.
* **Live Envelopes:** Generate moment, shear, normal force, and deformation envelopes.
* **Bridge Traffic Loads:** Includes moving vehicle loads with dynamic factor calculation.
* **Limit State Views:** Toggle between design and characteristic result combinations without re-running the solver.
* **Performance:** Uses a custom matrix stiffness solver accelerated with `numba`.

## Installation

### Prerequisites

* Python 3.10 or newer
* Git

### Get the Code

```bash
git clone https://github.com/KasperLFabricius/BriCoS.git
cd BriCoS
```

### Install and Run

```bash
pip install -r requirements.txt
streamlit run run_app.py
```

### Development Tests

```bash
pip install -r requirements-dev.txt
pytest
```

### Building a Standalone Executable

To package BriCoS as a Windows app that runs **without a Python install**, see
**[PACKAGING.md](PACKAGING.md)**. In short: run `build.bat` (or
`pyinstaller bricos.spec`), then distribute the resulting `dist\BriCoS` folder.
