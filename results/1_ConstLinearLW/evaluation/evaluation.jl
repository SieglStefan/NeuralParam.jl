# XXX


using Revise
using NeuralParam
using SpeedyWeather
using Plots
using Dates
using Accessors



TRUNC = 31
NLAYERS = 8


SIM_TIME = 3*31   # sampling time in days
SAMPLE_GAP = 1  # days between sampling

spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)


scheme_target = OneBandLongwave(spectral_grid)

run = "run_T31_L8_2026-07-07_23-22-44"
scheme_comp = load_scheme(
    path = joinpath(@__DIR__, "..", "calibration", run),
    file = "scheme.jld2"
)



schemes = (; 
    target = scheme_target, 
    comp = scheme_comp,
    none = nothing, 
    uncal = ConstLinearLW(spectral_grid)
)


data = sample_sims(
    spectral_grid,
    schemes;
    fac_pert_T = 2f0,
    fac_pert_q = 0f0,
    t_spinup = Day(31),
    sim_time = SIM_TIME,
    sample_gap = SAMPLE_GAP,
)



p_rmse = plot_comparison(
    data.trajectories.target,                            # Referenz = Wahrheit
    (; comp = data.trajectories.comp,
       none = data.trajectories.none,
       uncal = data.trajectories.uncal);
    metric = rmse, Δt_sample = data.Δt_sample,
    ylabel = "Temperature (K)")

p_bias = plot_comparison(
    data.trajectories.target,                            # Referenz = Wahrheit
    (; comp = data.trajectories.comp,
       none = data.trajectories.none,
       uncal = data.trajectories.uncal);
    metric = bias, Δt_sample = data.Δt_sample,
    ylabel = "Temperature (K)")

run_dir = joinpath(@__DIR__, run)
mkpath(run_dir)


display(p_rmse)
display(p_bias)

Plots.savefig(p_rmse, joinpath(run_dir, "rmse.png"))
Plots.savefig(p_bias, joinpath(run_dir, "bias.png"))


data2 = sample_sims(
    spectral_grid,
    (;target = scheme_target, comp = scheme_comp);
    fac_pert_T = 2f0,
    fac_pert_q = 0f0,
    t_spinup = Day(31),
    sim_time = 180+60,
    sample_gap = 3,
)


p_rmse1 = plot_comparison(
    data2.trajectories.target,                            # Referenz = Wahrheit
    (; comp = data2.trajectories.comp);
    metric = rmse, Δt_sample = data2.Δt_sample,
    ylabel = "Temperature (K)")

p_bias1 = plot_comparison(
    data2.trajectories.target,                            # Referenz = Wahrheit
    (; comp = data2.trajectories.comp);
    metric = bias, Δt_sample = data2.Δt_sample,
    ylabel = "Temperature (K)")


display(p_rmse1)
display(p_bias1)