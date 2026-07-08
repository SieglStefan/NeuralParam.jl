using Plots

run = "run_T31_L8_2026-07-07_23-22-44"
scheme_comp = load_scheme(
    path = joinpath(@__DIR__, "..", "calibration", run),
    file = "scheme.jld2"
)


a, b   = scheme_comp.ps.a, scheme_comp.ps.b
nlay   = length(a)
layers = 1:nlay
p_a = plot(layers, [fill(-1f0, nlay) a]; label=["init (−1)" "kalibriert"],
           ylabel="a", marker=:circle, lw=2)
p_b = plot(layers, [fill( 1f0, nlay) b]; label=["init (+1)" "kalibriert"],
           ylabel="b", xlabel="Layer", marker=:circle, lw=2)
plot(p_a, p_b; layout=(2,1), plot_title="Kalibriert vs. Init")