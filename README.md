# Hardware-Software Co-Design of Scalable, Energy-Efficient Analog Recurrent Computations

## What do you find in this repository?

This repository contains all code and data associated with **Hardware-Software Co-Design of Scalable, Energy-Efficient Analog Recurrent Computations**.

## How to use the code?

### Software (Section 2.3: First-quadrant BMRU & Section 4: Noise immunity)

Python with JAX/Flax was used for the software learning experiments (both the software and hardware backbones). All code is ready to run -- see the `software/` folder for details. Dependencies are listed in `requirements_frozen.txt`.

You can execute the different experiments by running the corresponding shell scripts in `experiments/`:
- `listops_shakespear.sh` — benchmark experiments (ListOps and Shakespeare language modeling, Table 1)
- `cmos_kws.sh` — benchmark experiments (sMNIST, pMNIST and dKWS, Table 1)
- `robustness_mismatch.sh` — large-scale noise robustness analysis (Section 4, Figure 3)

### Hardware (Section 3: Proof of concept: keyword spotting & Section 4: Power scaling)

Python 3.12 with JAX/Flax was used for binary keyword spotting training (with the hardware backbone, a specific folder had been made under `hardware/` for clarity). A minimal training and export pipeline is available under `hardware/learning/`. Dependencies are listed in `requirements.txt`. Shell scripts are provided for training, testing, and exporting.

### Cadence Post-Processing
Julia 1.11 was used for reading and plotting Cadence output logits and power consumption, as well as for computing transistor width lookup tables and producing all additional figures (data used for Monte Carlo and PVT analysis is not included in this repository due to its size). The code is located under `hardware/Cadence/`.

To use these scripts:
1. Download the latest version of Julia [here](https://julialang.org/) if not already installed.
2. Navigate to the folder containing `dependencies.jl` and run:
```julia
include("dependencies.jl")
```

### Launching Jupyter with Julia
To open Jupyter Notebook or JupyterLab using Julia:
```julia
using IJulia
notebook()  # or jupyterlab()
```
Then browse to the `.ipynb` files. If this is your first time launching Jupyter from Julia, you will be prompted to install it via Conda -- accept to proceed.

## Limited use license
**This software is not open source. It is protected by patents and is made available under a limited evaluation license only.**

You may use the software solely for testing and evaluation purposes. Any commercial use is prohibited. You may not copy, modify, reuse, integrate, or redistribute the code, in whole or in part, in any software or product that you distribute.

Please refer to the LICENSE file for the complete terms and conditions.
