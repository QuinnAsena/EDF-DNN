# EDF-DNN

Deep neural networks (MLP, CNN, U-Net) for mapping burn risk on a 1 km grid,
trained on historical drivers and projected onto CMIP6 future climate.

## Setup (WSL / Linux)

```bash
curl -fsSL https://pixi.sh/install.sh | sh      # once per machine
cd ~/projects/EDF-DNN
pixi install                                    # builds the CPU environment from pixi.lock
pixi run test                                   # checks everything imports and runs
pixi run register-kernel                        # Jupyter kernel for Quarto reports
```

On a GPU node (e.g. NCAR Casper): `pixi install -e cuda`, then `pixi run -e cuda ...`.

Input data are not in git; see `configs/data.yaml` for the expected files in `data/`.
