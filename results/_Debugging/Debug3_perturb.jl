### Debug3_perturb.jl — visual + quantitative test of perturb_grid_field!
###
### Fresh process:
###   julia --project=C:/Code/sw-ml-thesis-refactor
###   julia> include("results/_Debugging/Debug3_perturb.jl")
###
### Saves heatmap PNGs next to this file and prints:
###   - is the perturbed / spun-up state finite (no NaN)?
###   - perturbation magnitude (additive K for T, relative % for q)
###   - spatial decorrelation between perturbed and unperturbed spin-up

using Revise
using SpeedyWeather, NeuralParam
using CairoMakie, RingGrids
using Random, Statistics

const SG    = SpectralGrid(trunc = 31, nlayers = 8)
const KPLOT = SG.nlayers          # near-surface layer to visualise
const OUT   = @__DIR__

# --- one layer of a 3D grid field -> lon/lat matrix (adjust if your SW differs) ---
function layer_matrix(field, k)
    layer = field[:, k]
    g     = layer.grid
    full  = RingGrids.interpolate(RingGrids.full_grid_type(g), g.nlat_half, layer)
    return Matrix(full)
end

function save_heatmap(M, ttl, fname)
    fig = Figure()
    ax  = Axis(fig[1, 1]; title = ttl)
    hm  = heatmap!(ax, M)
    Colorbar(fig[1, 2], hm)
    CairoMakie.save(joinpath(OUT, fname), fig)
    @info "saved $fname"
end

# 1. base state (unperturbed), grid populated
model = PrimitiveWetModel(; spectral_grid = SG)
sim   = initialize!(model)
SpeedyWeather.initialize!(sim)                 # populate grid from spectral
T0 = copy(sim.variables.grid.temperature)
q0 = copy(sim.variables.grid.humidity)

# 2. perturb a COPY (keep `sim` as the unperturbed reference)
sim_pert = deepcopy(sim)
perturb_grid_field!(sim_pert, :temperature; fac_add  = 2f0)
perturb_grid_field!(sim_pert, :humidity;    fac_mult = 0.2f0, zeromin = true)
SpeedyWeather.initialize!(sim_pert)            # refresh grid to read the perturbation
Tp = copy(sim_pert.variables.grid.temperature)
qp = copy(sim_pert.variables.grid.humidity)

@info "perturbation finite?"  T = all(isfinite, Tp)  q = all(isfinite, qp)
@info "T perturbation (additive)"  max_dK   = maximum(abs.(Tp .- T0))                       T_range = extrema(Tp)
@info "q perturbation (relative)"  max_frac = maximum(abs.((qp .- q0) ./ max.(q0, 1f-9)))   q_range = extrema(qp)

# 3. heatmaps of the perturbation at layer KPLOT
try
    save_heatmap(layer_matrix(T0, KPLOT),        "T before",       "perturb_T_before.png")
    save_heatmap(layer_matrix(Tp, KPLOT),        "T after",        "perturb_T_after.png")
    save_heatmap(layer_matrix(Tp .- T0, KPLOT),  "T perturbation", "perturb_T_diff.png")
catch e
    @warn "heatmap failed — adjust layer_matrix() for your SW version" exception = e
end

# 4. spin up BOTH from the same base, compare (decorrelation)
run!(sim,      period = Day(7))                # unperturbed reference
run!(sim_pert, period = Day(7))                # perturbed
Tr = sim.variables.grid.temperature
Ts = sim_pert.variables.grid.temperature

@info "after spin-up finite?"  ref = all(isfinite, Tr)  pert = all(isfinite, Ts)
@info "decorrelation (1.0 = identical, lower = more diverged)"  spatial_cor = cor(vec(Tr), vec(Ts))

try
    save_heatmap(layer_matrix(Tr, KPLOT),       "spin-up unperturbed", "spinup_ref.png")
    save_heatmap(layer_matrix(Ts, KPLOT),       "spin-up perturbed",   "spinup_pert.png")
    save_heatmap(layer_matrix(Ts .- Tr, KPLOT), "spin-up difference",  "spinup_diff.png")
catch e
    @warn "heatmap failed" exception = e
end

@info "=== Debug3 done — check the .png files in $OUT ==="
