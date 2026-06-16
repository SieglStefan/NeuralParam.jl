### Debug1.jl — staged smoke tests for NeuralParam
###
### Run TOP TO BOTTOM. Each stage isolates one layer of the code, so the first
### stage that fails tells you where the problem is. Stages get progressively
### heavier (the Enzyme one is slow).
###
### NOTE: NLAYERS must match an existing data file:
###   data/zscore/llw_L<NLAYERS>.jld2  and  data/scaling/llw_L<NLAYERS>.jld2
### Change NLAYERS below to whatever you actually have.

using Revise
using NeuralParam
using SpeedyWeather
using Lux
using Random
using Test

const TRUNC   = 31
const NLAYERS = 8        # <-- must match your llw_L<NLAYERS>.jld2 data files

spectral_grid = SpectralGrid(trunc = TRUNC, nlayers = NLAYERS)


# =====================================================================
@info "STAGE 1 — build configs and parameterizations"
# Catches: config constructors, data-file loading (zscore + scaling),
#          NN setup, struct construction.
# =====================================================================

mlp = MLPConfig(n_hidden = 2, width = 16)

# ConstLinearLW (no data files needed)
const_cfg = ConstLinearLWConfig(spectral_grid; name = "debug_const")
const_lw  = ConstLinearLW(const_cfg)
@test const_lw.ps.a isa Vector{Float32}
@test length(const_lw.ps.a) == NLAYERS

# NeuralLinearLW (loads llw_L<NLAYERS>.jld2 from data/zscore and data/scaling)
nlin_cfg = NeuralLinearLWConfig(spectral_grid, mlp; name = "debug_nlin")
nlin_lw  = NeuralLinearLW(nlin_cfg)
@test nlin_cfg.n_in  == NLAYERS
@test nlin_cfg.n_out == 2 * NLAYERS
@test length(nlin_lw.input_buffer) == nlin_cfg.n_in

@info "STAGE 1 passed ✓"


# =====================================================================
@info "STAGE 2 — utility functions"
# Catches: zscore math, metrics, tree norms.
# =====================================================================

@test inv_zscore(zscore(5f0, 2f0, 3f0), 2f0, 3f0) ≈ 5f0          # roundtrip
@test rmse([1f0, 2f0], [1f0, 2f0]) == 0f0
@test rmse([0f0, 0f0], [3f0, 4f0]) ≈ sqrt((9 + 16) / 2)
@test tree_l2norm((; a = [3f0], b = [4f0])) ≈ 5f0                # 3-4-5

@info "STAGE 2 passed ✓"


# =====================================================================
@info "STAGE 3 — forward pass inside a real SpeedyWeather run"
# Catches: parameterization! bugs (indexing, buffer, NaNs), bad scaling,
#          the @propagate_inbounds loop. If T blows up to NaN/Inf the
#          parameterization is unstable or mis-indexed.
# =====================================================================

for (name, lw) in (("ConstLinearLW", const_lw), ("NeuralLinearLW", nlin_lw))
    model = PrimitiveWetModel(; spectral_grid, longwave_radiation = lw)
    sim   = initialize!(model)
    run!(sim, period = Day(1))
    T = sim.variables.grid.temperature
    @info "  $name ran" extrema_T = extrema(T)
    @test all(isfinite, T)
end

@info "STAGE 3 passed ✓"


# =====================================================================
@info "STAGE 4 — perturbation helper"
# Catches: the dynamic-keyword set! in perturb_grid_field!, double-scaling,
#          NaNs. Confirms the field actually changes and stays finite.
# =====================================================================

model = PrimitiveWetModel(; spectral_grid)
sim   = initialize!(model)
T_before = copy(sim.variables.grid.temperature)
perturb_grid_field!(sim, :temperature, fac_add = 2f0)
T_after = sim.variables.grid.temperature
@test all(isfinite, T_after)
@test T_before != T_after          # something actually changed
@info "STAGE 4 passed ✓" maxchange = maximum(abs.(T_after .- T_before))


# =====================================================================
@info "STAGE 5 — training loop WITHOUT autodiff (test_mode = true)"
# Catches: spinup, implicit warm-up, copy!, sim_timesteps!, @set swap,
#          logging plumbing, run_training return signature.
# Gradients are zero here, so loss should NOT move — we only check it RUNS.
# =====================================================================

cfg = TrainingConfig(Val(:test))   # 1 IC, 1 traj, 1 epoch, n_steps = 1

out = run_training(nlin_lw, spectral_grid; training_config = cfg, test_mode = true)
@info "STAGE 5 passed ✓ — run_training returned $(length(out)) values"


# =====================================================================
@info "STAGE 6 — training loop WITH autodiff (slow: Enzyme compiles)"
# Catches: Enzyme differentiating timestep!/parameterization!, gradient
#          extraction, AND the seed sign — loss should DECREASE.
# Bump epochs so a trend is visible.
# =====================================================================

cfg_ad = TrainingConfig(
    n_ic = 1, n_traj = 1, n_epochs = 10, n_steps = 1, n_gap = 1,
    t_spinup = Day(1),
)

out_ad = run_training(nlin_lw, spectral_grid; training_config = cfg_ad, test_mode = false)
# Pull the loss vector out of the return (index depends on your signature):
#   if run_training returns (lw, L, PN, GN)  ->  L = out_ad[2]
L = out_ad[2]
@info "STAGE 6 — loss trajectory" first = L[1] last = L[end]
@test all(isfinite, L)
@test L[end] < L[1]   # <-- if this FAILS, the seed_loss! sign is likely flipped

@info "STAGE 6 passed ✓"


# =====================================================================
@info "STAGE 7 — save / load roundtrip"
# Catches: io.jl for the const (no st) and neural (with st) cases.
# =====================================================================

tmp = mktempdir()
for lw in (const_lw, nlin_lw)
    path = save_longwave(; path = tmp, radiation = lw)
    back = load_longwave(; path = tmp, name = lw.config.name)
    @test back.config.name == lw.config.name
    @info "  saved+loaded $(lw.config.name) ✓"
end

@info "ALL STAGES PASSED ✓✓✓"
