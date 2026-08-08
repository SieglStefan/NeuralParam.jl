### Inspect a stored rollout: sanity checks, metric curves, heatmaps

using Revise
using NeuralParam
using CairoMakie


NAME  = "TEST_ROLLOUT"
DIR   = rollout_dir(NAME)
PLOTS = joinpath(DIR, "plots")
mkpath(PLOTS)


### Load
rollouts = collect_rollouts([NAME])
r = rollouts[Symbol(NAME)]

@show r.days r.probes r.start_days r.heatmap_days r.heatmap_traj


### Sanity check: at lead day 0 the rollout IS the reference state
#   -> must be exactly zero, otherwise copy!/restart disagree with the stored reference
if first(r.heatmap_days) == 0
    for p in r.probes
        d = maximum(abs, r.heatmap_states[p][1] .- r.heatmap_ref[p][1])
        println("day-0  max|rollout - ref|  $(rpad(p, 5)): $d")
    end
end


### Metric curves — one .png per metric into DIR/plots
figs = plot_rollout_metrics(; rollouts, dir = PLOTS, probes = (:T, :olw, :slwd, :slwu))
figs.rmse       # display in the REPL; also figs.bias, figs.maxdiff, figs.mean


### Raw numbers behind the curves (vector over lead days)
@show rollout_curve(r, :T,    :rmse)            # all layers, combined in quadrature
@show rollout_curve(r, :T,    :rmse; layer = 8) # bottom layer only
@show rollout_curve(r, :olw,  :bias)
@show rollout_curve(r, :slwd, :bias)
@show rollout_curve(r, :T,    :mean)            # drift of the rollout itself


### Heatmaps — one figure per heatmap day
hm_T    = plot_rollout_heatmaps(; rollouts, var = :T, layer = 8)
hm_olw  = plot_rollout_heatmaps(; rollouts, var = :olw)
hm_slwd = plot_rollout_heatmaps(; rollouts, var = :slwd)

for (j, d) in enumerate(r.heatmap_days)
    CairoMakie.save(joinpath(PLOTS, "hm_T_L8_day$(d).png"), hm_T[j])
    CairoMakie.save(joinpath(PLOTS, "hm_olw_day$(d).png"),  hm_olw[j])
    CairoMakie.save(joinpath(PLOTS, "hm_slwd_day$(d).png"), hm_slwd[j])
end