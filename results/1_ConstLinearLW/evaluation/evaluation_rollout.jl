# XXX


using Revise
using NeuralParam
using SpeedyWeather
using Plots
using Dates
using Accessors



TRUNC = 31
NLAYERS = 8

SAMPLE_RES = 1


spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)


run_const = "run_T31_L8_2026-07-07_23-22-44"
scheme_const = load_scheme(
    path = joinpath(@__DIR__, "..", "calibration", run_const),
    file = "scheme.jld2"
)

run_neural = "run_T31_L8_2026-07-07_23-22-44"
scheme_neural = load_scheme(
    path = joinpath(@__DIR__, "..", "calibration", run_neural),
    file = "scheme.jld2"
)

schemes = (; 
    target = OneBandLongwave(spectral_grid), 
    comp = scheme_comp,
    none = nothing, 
    uncal = ConstLinearLW(spectral_grid)
)


data_comp = sample_sims(
    spectral_grid,
    schemes;
    fac_pert_T = 2f0,
    fac_pert_q = 0f0,
    t_spinup = Day(31),
    sim_time = 90,
    sample_gap = SAMPLE_RES,
)



p_comp_rmse = plot_comparison(
    data_comp.trajectories.target,
    (; comp = data_comp.trajectories.comp,
       none = data_comp.trajectories.none,
       uncal = data_comp.trajectories.uncal);
    metric = rmse, Δt_sample = data_comp.Δt_sample,
    ylabel = "Temperature (K)"
)

p_comp_bias = plot_comparison(
    data_comp.trajectories.target,
    (; comp = data_comp.trajectories.comp,
       none = data_comp.trajectories.none,
       uncal = data_comp.trajectories.uncal);
    metric = bias, Δt_sample = data_comp.Δt_sample,
    ylabel = "Temperature (K)"
)



titles = ["target / OneBandLongwave", "comp / ConstLinearLW","none / No LW Parameterization","uncal / Uncalibrated ConstLinearLW"] 
heatmap_comp_0 = plot_heatmaps_eval(data_comp.trajectories, data_comp.Δt_sample, 0;  layer=8, titles=TITLES)
heatmap_comp_7 = plot_heatmaps_eval(data_comp.trajectories, data_comp.Δt_sample, 7;  layer=8, titles=TITLES)
heatmap_comp_30 = plot_heatmaps_eval(data_comp.trajectories, data_comp.Δt_sample, 30;  layer=8, titles=TITLES)
heatmap_comp_90 = plot_heatmaps_eval(data_comp.trajectories, data_comp.Δt_sample, 90;  layer=8, titles=TITLES)


run_dir = joinpath(@__DIR__, run)
mkpath(run_dir)
mkpath(joinpath(run_dir, "plots_rollout"))


Plots.savefig(p_comp_rmse, joinpath(run_dir, "plots_rollout", "comp_timeseries_rmse.png"))
Plots.savefig(p_comp_bias, joinpath(run_dir, "plots_rollout", "comp_timeseries_bias.png"))

CairoMakie.save(joinpath(run_dir, "plots_rollout", "comp_heatmap_0.png"), heatmap_comp_0)
CairoMakie.save(joinpath(run_dir, "plots_rollout", "comp_heatmap_7.png"), heatmap_comp_7)
CairoMakie.save(joinpath(run_dir, "plots_rollout", "comp_heatmap_30.png"), heatmap_comp_30)
CairoMakie.save(joinpath(run_dir, "plots_rollout", "comp_heatmap_90.png"), heatmap_comp_90)






data_emulator = sample_sims(
    spectral_grid,
    (;target = OneBandLongwave(spectral_grid), comp = scheme_comp);
    fac_pert_T = 2f0,
    fac_pert_q = 0f0,
    t_spinup = Day(31),
    sim_time = 180,
    sample_gap = SAMPLE_RES,
)


p_emulator_rmse = plot_comparison(
    data_emulator.trajectories.target,
    (; comp = data_emulator.trajectories.comp);
    metric = rmse, Δt_sample = data_emulator.Δt_sample,
    ylabel = "Temperature (K)"
)

p_emulator_bias = plot_comparison(
    data_emulator.trajectories.target,
    (; comp = data_emulator.trajectories.comp);
    metric = bias, Δt_sample = data_emulator.Δt_sample,
    ylabel = "Temperature (K)"
)

titles = ["target / OneBandLongwave", "comp / ConstLinearLW"] 
heatmap_emulator_0 = plot_heatmaps_eval(data_emulator.trajectories, data_emulator.Δt_sample, 0;  layer=8, titles=TITLES)
heatmap_emulator_7 = plot_heatmaps_eval(data_emulator.trajectories, data_emulator.Δt_sample, 7;  layer=8, titles=TITLES)
heatmap_emulator_30 = plot_heatmaps_eval(data_emulator.trajectories, data_emulator.Δt_sample, 30;  layer=8, titles=TITLES)
heatmap_emulator_90 = plot_heatmaps_eval(data_emulator.trajectories, data_emulator.Δt_sample, 90;  layer=8, titles=TITLES)




Plots.savefig(p_emulator_rmse, joinpath(run_dir, "plots_rollout", "emulator_timeseries_rmse.png"))
Plots.savefig(p_emulator_bias, joinpath(run_dir, "plots_rollout", "emulator_timeseries_bias.png"))

CairoMakie.save(joinpath(run_dir, "plots_rollout", "emulator_heatmap_0.png"), heatmap_emulator_0)
CairoMakie.save(joinpath(run_dir, "plots_rollout", "emulator_heatmap_7.png"), heatmap_emulator_7)
CairoMakie.save(joinpath(run_dir, "plots_rollout", "emulator_heatmap_30.png"), heatmap_emulator_30)
CairoMakie.save(joinpath(run_dir, "plots_rollout", "emulator_heatmap_90.png"), heatmap_emulator_90)
