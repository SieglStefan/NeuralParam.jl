using SpeedyWeather
using Dates
using Enzyme

# Assumption: dein Modul ist schon geladen:
# include(joinpath(@__DIR__, "..", "src", "NeuralLongwave.jl"))
# using .NeuralLongwave

# Minimal grid: möglichst klein, damit Enzyme/Lux nicht direkt explodiert
spectral_grid = SpectralGrid(trunc=15, nlayers=1)

# Minimal NN: keine hidden layer, nur linearer Dense von 1 -> 2
config = NeuralLinearLongwaveConfig(
    width = 2,
    n_hidden = 0,
)

radiation_nllw = NeuralLinearLongwave(spectral_grid; config)

# Template erstellen
sim_template = NeuralLongwave.create_template(spectral_grid)

# Target/train/reference Sims erzeugen
sim_ref, sim_target, sim_train = NeuralLongwave.create_sims(
    spectral_grid,
    sim_template;
    radiation = radiation_nllw,
    t_spinup = Day(1),
)

n_steps = 1

# Beide Sims korrekt initialisieren und ersten Leapfrog-Start machen
vars0 = NeuralLongwave.prepare_sim_pair!(sim_target, sim_train, n_steps)

# Target final erzeugen
NeuralLongwave.sim_timesteps!(sim_target, n_steps)

# Train final erzeugen
NeuralLongwave.reset_sim!(sim_train, vars0)
NeuralLongwave.sim_timesteps!(sim_train, n_steps)

@info "VOR COMPUTE_GRADIENTS"

vars_ad = deepcopy(vars0)

bvars_ad = make_zero(vars_ad)
T_target = sim_target.variables.grid.temperature
T_train  = sim_train.variables.grid.temperature
N = length(T_train)
bvars_ad.grid.temperature .= 2 .* (T_train .- T_target) ./ N

model_ad = deepcopy(sim_train.model)
bmodel_ad = make_zero(model_ad)

@info "VOR AD OHNE CHECKPOINTING"


function one_timestep!(vars_ad, model_ad)
    SpeedyWeather.timestep!(vars_ad, 2model_ad.time_stepping.Δt, model_ad, 2, 2)
    return nothing
end


@time Enzyme.autodiff(
    Enzyme.Reverse,
    one_timestep!,
    Enzyme.Const,
    Enzyme.Duplicated(vars_ad, bvars_ad),
    Enzyme.Duplicated(model_ad, bmodel_ad),
)

@info "AD OHNE CHECKPOINTING OK"
@show NeuralLongwave.tree_l2norm(bmodel_ad.longwave_radiation.ps)