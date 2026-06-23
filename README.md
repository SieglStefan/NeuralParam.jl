# NeuralParam.jl

**Learning longwave-radiation parameterizations for [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) by differentiable online training.**

> 🎓 This repository contains the code for my **Master's thesis** on data-driven
> neural parameterizations for SpeedyWeather.jl, written in the
> **M.Sc. Theoretical and Mathematical Physics (TMP)** program jointly run by
> **LMU Munich** and **TU Munich**.

## What it does

NeuralParam.jl replaces a physical longwave-radiation parameterization in the
[SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) atmospheric
model with a small neural network and trains it **online**: gradients of a
*trajectory-matching loss* are propagated *through* the model's time stepping with
reverse-mode automatic differentiation
([Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) +
[Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl)),
so the scheme is optimized in the same dynamical setting in which it later runs.

The loss is the temperature RMSE between a *target* and a *trained* simulation after a
short trajectory segment, averaged over many perturbed initial conditions.

## Schemes

| Scheme | Type | Idea |
| --- | --- | --- |
| `ConstLinearLW` | baseline | Global per-layer constants `a, b` in `dT = a·T + b` (linearized Stefan–Boltzmann / Budyko–Sellers). |
| `NeuralLinearLW` | neural, linear | Neural network predicts `a, b` per layer. |
| `NeuralABRLW` | neural, column | Emulates `AnalyticBandRadiation.jl` column-wise. |
| `NeuralABRLWGlobal` | neural, global | Grid-wide variant of the ABR emulator, designed for GPU execution. |

Network architectures (`MLPConfig`, …) are pluggable via the `architectures/` module.

## Repository layout

```
src/
  architectures/      # neural network configs (MLP, RNN)
  parameterizations/  # longwave radiation schemes (linear, ABR)
  training/           # online training loop, gradients, run config
  utils/              # stats (z-score, scaling), io, metrics, plotting
scripts/stats/        # precompute normalization statistics -> data/stats/*.jld2
results/              # numbered experiments (calibration, stability, ...)
tests/                # test suite
```

## Getting started

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()   # reproduces the pinned environment (Manifest.toml)

using SpeedyWeather, NeuralParam

spectral_grid = SpectralGrid(trunc=31, nlayers=8)

# neural longwave scheme + its network architecture
scheme = NeuralLinearLW(spectral_grid, MLPConfig(n_hidden=2, width=16))

# online training against SpeedyWeather's default longwave radiation
# returns: trained scheme, loss (L), parameter-norm (PN) and gradient-norm (GN) histories
param, L, PN, GN = run_training(spectral_grid, scheme)
```

Normalization statistics in `data/stats/` are produced by the scripts in
`scripts/stats/` and loaded automatically by the neural schemes.

## License

Released under the MIT License.
