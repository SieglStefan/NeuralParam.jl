using SpeedyWeather
using Enzyme
using Dates
using Random
using Lux

# Assumption:
# NeuralLinearLongwaveConfig is already loaded before this file.
# Otherwise uncomment/adapt:
# include(joinpath(@__DIR__, "..", "src", "NeuralLongwave.jl"))
# using .NeuralLongwave

# ------------------------------------------------------------
# Test parameterization:
# Lux-like ps, but scalar allocation-free generated forward.
#
# Test case:
# nlayers = 1
# arbitrary width
# n_hidden = 0
#
# NN architecture:
# Dense(1 => width, tanh)
# Dense(width => 2)
# ------------------------------------------------------------

@kwdef mutable struct ManualNLLWGeneratedTest{M,P,S,C} <: SpeedyWeather.AbstractLongwave
    nn::M
    ps::P
    st::S
    config::C
end

function ManualNLLWGeneratedTest(
    SG::SpeedyWeather.SpectralGrid;
    config = NeuralLinearLongwaveConfig(width = 4, n_hidden = 0),
    rng = Random.default_rng(),
)
    n_in = SG.nlayers
    n_out = 2n_in
    width = config.width

    @assert n_in == 1 "This test only supports nlayers = 1"
    @assert config.n_hidden == 0 "This test only supports n_hidden = 0"

    nn = Lux.Chain(
        Lux.Dense(n_in => width, tanh),
        Lux.Dense(width => n_out),
    )

    ps, st = Lux.setup(rng, nn)
    st = Lux.testmode(st)

    return ManualNLLWGeneratedTest(; nn, ps, st, config)
end

function SpeedyWeather.initialize!(::ManualNLLWGeneratedTest, ::AbstractModel)
    return nothing
end

# ------------------------------------------------------------
# Generated scalar forward
#
# This generates code equivalent to:
#
# h1 = tanh(W1[1,1] * x + b1[1])
# h2 = tanh(W1[2,1] * x + b1[2])
# ...
# y1 = b2[1] + sum(W2[1,j] * hj)
# y2 = b2[2] + sum(W2[2,j] * hj)
#
# No arrays, no zeros, no broadcast, no matrix multiplication.
# ------------------------------------------------------------

@generated function generated_mlp_forward_1d(x, ps, ::Val{W}) where {W}
    h_syms = [Symbol(:h, j) for j in 1:W]

    h_lines = [
        :($(h_syms[j]) = tanh(ps.layer_1.weight[$j, 1] * x + ps.layer_1.bias[$j]))
        for j in 1:W
    ]

    y1_terms = [:(ps.layer_2.weight[1, $j] * $(h_syms[j])) for j in 1:W]
    y2_terms = [:(ps.layer_2.weight[2, $j] * $(h_syms[j])) for j in 1:W]

    y1_expr = :(ps.layer_2.bias[1])
    y2_expr = :(ps.layer_2.bias[2])

    for term in y1_terms
        y1_expr = :($y1_expr + $term)
    end

    for term in y2_terms
        y2_expr = :($y2_expr + $term)
    end

    return quote
        $(h_lines...)
        y1 = $y1_expr
        y2 = $y2_expr
        return y1, y2
    end
end

# ------------------------------------------------------------
# Parameterization
# ------------------------------------------------------------

Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij, vars::Variables, para::ManualNLLWGeneratedTest, model::AbstractModel
)
    (; ps, config) = para

    # nlayers = 1 only for this test
    Tk = vars.grid.temperature[ij, 1]
    x = (Tk - config.T_mean) / config.T_std

    y1, y2 = generated_mlp_forward_1d(x, ps, Val(config.width))

    ak = config.sc_a * y1
    bk = config.sc_b * y2

    vars.tendencies.grid.temperature[ij, 1] += ak * Tk + bk

    return nothing
end

# ------------------------------------------------------------
# Helpers
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

rng = Random.default_rng()
Random.seed!(rng, 1234)

spectral_grid = SpectralGrid(trunc = 15, nlayers = 1)

config = NeuralLinearLongwaveConfig(
    width = 4,
    n_hidden = 0,
)

radiation_nllw = ManualNLLWGeneratedTest(spectral_grid; config, rng)

@info "ps structure"
@show typeof(radiation_nllw.ps)
@show keys(radiation_nllw.ps)
@show size(radiation_nllw.ps.layer_1.weight)
@show size(radiation_nllw.ps.layer_2.weight)

# Forward equivalence check against Lux.apply for one scalar input
x_test = Float32[250.0]
y_lux, _ = Lux.apply(radiation_nllw.nn, x_test, radiation_nllw.ps, radiation_nllw.st)
y_gen = collect(generated_mlp_forward_1d(x_test[1], radiation_nllw.ps, Val(config.width)))

@info "Forward sanity check"
@show y_lux
@show y_gen
@show maximum(abs, y_lux .- y_gen)

model_target = PrimitiveWetModel(; spectral_grid)
sim_target = initialize!(model_target)

model_train = PrimitiveWetModel(; spectral_grid, longwave_radiation = radiation_nllw)
sim_train = initialize!(model_train)

# Put both simulations into same initial state
copy!(sim_train.variables, sim_target.variables)

# Initialize leapfrog/startup consistently
SpeedyWeather.initialize!(sim_target, steps = 2)
SpeedyWeather.initialize!(sim_train, steps = 2)

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

# Seed MSE gradient wrt final temperature
bvars_ad.grid.temperature .= 2 .* (T_train .- T_target) ./ N

model_ad = deepcopy(sim_train.model)
bmodel_ad = Enzyme.make_zero(model_ad)

@show tree_l2norm(bmodel_ad.longwave_radiation.ps)

# ------------------------------------------------------------
# Actual AD test
# ------------------------------------------------------------

@info "Starting Enzyme autodiff with generated scalar forward"

@time Enzyme.autodiff(
    Enzyme.Reverse,
    one_timestep!,
    Enzyme.Const,
    Enzyme.Duplicated(vars_ad, bvars_ad),
    Enzyme.Duplicated(model_ad, bmodel_ad),
)

@info "AD finished"

@show tree_l2norm(bmodel_ad.longwave_radiation.ps)

@show maximum(abs, bmodel_ad.longwave_radiation.ps.layer_1.weight)
@show maximum(abs, bmodel_ad.longwave_radiation.ps.layer_1.bias)
@show maximum(abs, bmodel_ad.longwave_radiation.ps.layer_2.weight)
@show maximum(abs, bmodel_ad.longwave_radiation.ps.layer_2.bias)

@info "Script finished"