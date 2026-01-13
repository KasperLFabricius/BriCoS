# BriCoS (Bridge Comparison Software)

**BriCoS** is a lightweight, browser-based Finite Element Method (FEM) tool designed specifically for structural engineers. It allows for the rapid, simultaneous comparison of two 2D bridge systems ("System A" and "System B") with a focus on speed and adherence to Danish bridge standards.

## 🚀 Key Features

* **Dual System Analysis:** Model, analyze, and visualize two structural systems side-by-side to instantly compare static schemes or geometry options.
* **Live Envelopes:** Real-time generation of Moment ($M$), Shear ($V$), Normal ($N$), and Deformation ($\delta$) envelopes.
* **Eurocode Compliant:** Built-in load models according to **DS/EN 1991-2** (Danish Annex), including automatic Dynamic Factor ($\Phi$) calculation.
* **Instant Limit States:** Toggle between **ULS** (Design) and **SLS** (Characteristic) instantly without re-running the solver.
* **Performance:** Powered by a custom matrix stiffness solver accelerated with `numba`, capable of processing thousands of vehicle steps in seconds.

## 🛠️ Installation & Setup

### Prerequisites
* [Python 3.8+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads) (optional, if cloning)

### 1. Get the Code
Clone this repository or download the files to a local folder:
```bash
git clone [https://github.com/KasperLFabricius/BriCoS_.git](https://github.com/KasperLFabricius/BriCoS_.git)
cd BriCoS