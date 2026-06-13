using SpeedyWeather
using Enzyme
using Dates

# Assumption: NeuralLongwave module is already loaded before this file,
# or uncomment/adapt these lines:
# include(joinpath(@__DIR__, "..", "src", "NeuralLongwave.jl"))
# using .NeuralLongwave

# ------------------------------------------------------------
# Test parameterization:
# nn and st are stored globally, NOT inside the differentiated model.
# Only ps and config are fields of NeuralLinearLongwaveTest.
# ------------------------------------------------------------


@kwdef mutable struct ManualNLLWTest{P,C} <: SpeedyWeather.AbstractLongwave
    ps::P
    config::C
end

function ManualNLLWTest(
    SG::SpeedyWeather.SpectralGrid;
    config = NeuralLinearLongwaveConfig(),
    width = 4,
)
    ps = (
        W1 = 0.01f0 .* randn(Float32, width, 1),
        b1 = zeros(Float32, width),
        W2 = 0.01f0 .* randn(Float32, 2, width),
        b2 = zeros(Float32, 2),
    )

    return ManualNLLWTest(; ps, config)
end

function SpeedyWeather.initialize!(::ManualNLLWTest, ::AbstractModel)
    return nothing
end

Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij, vars::Variables, para::ManualNLLWTest, model::AbstractModel
)
    Tk = vars.grid.temperature[ij,1]
    x = (Tk - para.config.T_mean) / para.config.T_std

    h1 = tanh(para.ps.W1[1,1] * x + para.ps.b1[1])
    h2 = tanh(para.ps.W1[2,1] * x + para.ps.b1[2])
    h3 = tanh(para.ps.W1[3,1] * x + para.ps.b1[3])
    h4 = tanh(para.ps.W1[4,1] * x + para.ps.b1[4])

    y1 = para.ps.W2[1,1]*h1 + para.ps.W2[1,2]*h2 + para.ps.W2[1,3]*h3 + para.ps.W2[1,4]*h4 + para.ps.b2[1]
    y2 = para.ps.W2[2,1]*h1 + para.ps.W2[2,2]*h2 + para.ps.W2[2,3]*h3 + para.ps.W2[2,4]*h4 + para.ps.b2[2]

    ak = para.config.sc_a * y1
    bk = para.config.sc_b * y2

    vars.tendencies.grid.temperature[ij,1] += ak * Tk + bk

    return nothing
end

# ------------------------------------------------------------
# Minimal helpers for this local test
# ------------------------------------------------------------

tree_l2sum(x::Number) = abs2(x)
tree_l2sum(x::AbstractArray) = sum(abs2, x)
tree_l2sum(x::NamedTuple) = sum(tree_l2sum, values(x))
tree_l2sum(x::Tuple) = sum(tree_l2sum, x)
tree_l2norm(x) = sqrt(tree_l2sum(x))

rmse(x, y) = sqrt(sum((y .- x).^2) / length(x))

function one_timestep!(vars_ad, model_ad)
    SpeedyWeather.timestep!(
        vars_ad,
        2model_ad.time_stepping.Δt,
        model_ad,
        2,
        2,
    )
    return nothing
end

# ------------------------------------------------------------
# Build minimal simulation
# ------------------------------------------------------------

@info "Building minimal setup"

spectral_grid = SpectralGrid(trunc=15, nlayers=1)

config = NeuralLinearLongwaveConfig(
    width = 2,
    n_hidden = 0,
)

radiation_nllw = ManualNLLWTest(spectral_grid; config, width=4)

model_target = PrimitiveWetModel(; spectral_grid)
sim_target = initialize!(model_target)

model_train = PrimitiveWetModel(; spectral_grid, longwave_radiation = radiation_nllw)
sim_train = initialize!(model_train)

# Put both simulations into same initial state
copy!(sim_train.variables, sim_target.variables)

# Initialize leapfrog/startup consistently
SpeedyWeather.initialize!(sim_target, steps=2)
SpeedyWeather.initialize!(sim_train, steps=2)

SpeedyWeather.first_timesteps!(sim_target)
SpeedyWeather.first_timesteps!(sim_train)

vars0 = deepcopy(sim_train.variables)

# Create final target/train states for the loss seed
dt = 2sim_train.model.time_stepping.Δt

SpeedyWeather.timestep!(sim_target.variables, dt, sim_target.model)
SpeedyWeather.timestep!(sim_train.variables, dt, sim_train.model)

T_target = sim_target.variables.grid.temperature
T_train = sim_train.variables.grid.temperature
N = length(T_train)

@show rmse(T_train, T_target)

# ------------------------------------------------------------
# Prepare AD variables
# ------------------------------------------------------------

@info "Preparing AD objects"

vars_ad = deepcopy(vars0)
bvars_ad = Enzyme.make_zero(vars_ad)
bvars_ad.grid.temperature .= 2 .* (T_train .- T_target) ./ N

model_ad = deepcopy(sim_train.model)
bmodel_ad = Enzyme.make_zero(model_ad)

@show tree_l2norm(bmodel_ad.longwave_radiation.ps)

# ------------------------------------------------------------
# Actual AD test
# ------------------------------------------------------------

@info "Starting Enzyme autodiff without checkpointing"

@time Enzyme.autodiff(
    Enzyme.Reverse,
    one_timestep!,
    Enzyme.Const,
    Enzyme.Duplicated(vars_ad, bvars_ad),
    Enzyme.Duplicated(model_ad, bmodel_ad),
)

@info "AD finished"

@show tree_l2norm(bmodel_ad.longwave_radiation.ps)