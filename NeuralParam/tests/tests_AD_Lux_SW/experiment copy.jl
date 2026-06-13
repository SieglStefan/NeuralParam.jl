using SpeedyWeather
using Dates

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

# Minimaler AD-Test: genau ein compute_gradients-Aufruf
@time loss, grads = NeuralLongwave.compute_gradients(
    vars0,
    sim_target,
    sim_train,
    n_steps,
    false,   # false = echter Enzyme.autodiff-Aufruf
)

@show loss
@show NeuralLongwave.tree_l2norm(grads)