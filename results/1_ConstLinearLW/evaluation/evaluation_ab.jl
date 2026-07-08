using Plots
using NeuralParam


run = "run_T31_L8_2026-07-07_23-22-44"
scheme_comp = load_scheme(
    path = joinpath(@__DIR__, "..", "calibration", run),
    file = "scheme.jld2"
)

a, b   = scheme_comp.ps.a, scheme_comp.ps.b
nlay   = length(a)
layers = 1:nlay

p_a = plot([fill(-1f0, nlay) a], layers;                 # Werte auf x, Layer auf y
           label  = ["init (−1)" "calibrated"],
           xlabel = "a", ylabel = "Layer",
           title  = "Scaling parameter a",
           yflip  = true,                                 # Layer 1 oben (wie yreversed)
           yticks = layers, marker = :circle, lw = 2,
           legend =  :topleft)

p_b = plot([fill(1f0, nlay) b], layers;
           label  = ["init (+1)" "calibrated"],
           xlabel = "b", ylabel = "Layer",
           title  = "Scaling parameter b",
           yflip  = true,
           yticks = layers, marker = :circle, lw = 2,
           legend =  :topleft)

fig = plot(p_a, p_b; layout = (1, 2))   # a und b nebeneinander

# speichern
run_dir = joinpath(@__DIR__, run)
mkpath(run_dir)
mkpath(joinpath(run_dir, "plots_ab"))


Plots.savefig(fig, joinpath(run_dir, "plots_ab", "ab_parameters.png"))