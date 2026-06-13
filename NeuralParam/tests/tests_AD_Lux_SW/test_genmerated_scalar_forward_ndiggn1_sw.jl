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
# width arbitrary
# n_hidden = 1
#
# NN architecture:
# Dense(1 => width, tanh)
# Dense(width => width, tanh)
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
    config = NeuralLinearLongwaveConfig(width = 4, n_hidden = 1),
    rng = Random.default_rng(),
)
    n_in = SG.nlayers
    n_out = 2n_in
    width = config.width
    n_hidden = config.n_hidden

    @assert n_in == 1 "This test only supports nlayers = 1"
    @assert n_hidden == 1 "This test is specifically for n_hidden = 1"

    hidden = [Lux.Dense(width => width, tanh) for _ in 1:n_hidden]

    nn = Lux.Chain(
        Lux.Dense(n_in => width, tanh),
        hidden...,
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
# Generated scalar forward for nlayers = 1, arbitrary width W,
# and arbitrary n_hidden H.
#
# Layer convention:
# layer_1: input -> hidden
# layer_2 ... layer_(H+1): hidden -> hidden
# layer_(H+2): hidden -> output
#
# No arrays, no zeros, no broadcast, no matrix multiplication.
# ------------------------------------------------------------

@generated function generated_mlp_forward_1d(x, ps, ::Val{W}, ::Val{H}) where {W,H}
    lines = Expr[]

    # First layer: input scalar x -> hidden vector h_0_j
    prev = [Symbol(:h0_, j) for j in 1:W]

    for j in 1:W
        push!(
            lines,
            :($(prev[j]) = tanh(ps.layer_1.weight[$j, 1] * x + ps.layer_1.bias[$j]))
        )
    end

    # Hidden layers: hidden -> hidden
    # For H = 1, this creates layer_2.
    for ℓ in 1:H
        layer_idx = ℓ + 1
        curr = [Symbol(:h, ℓ, :_, j) for j in 1:W]

        for j in 1:W
            expr = :(ps.$(Symbol(:layer_, layer_idx)).bias[$j])

            for i in 1:W
                expr = :($expr + ps.$(Symbol(:layer_, layer_idx)).weight[$j, $i] * $(prev[i]))
            end

            push!(lines, :($(curr[j]) = tanh($expr)))
        end

        prev = curr
    end

    # Output layer
    out_layer_idx = H + 2

    y1 = :(ps.$(Symbol(:layer_, out_layer_idx)).bias[1])
    y2 = :(ps.$(Symbol(:layer_, out_layer_idx)).bias[2])

    for j in 1:W
        y1 = :($y1 + ps.$(Symbol(:layer_, out_layer_idx)).weight[1, $j] * $(prev[j]))
        y2 = :($y2 + ps.$(Symbol(:layer_, out_layer_idx)).weight[2, $j] * $(prev[j]))
    end

    push!(lines, :(y1 = $y1))
    push!(lines, :(y2 = $y2))
    push!(lines, :(return y1, y2))

    return Expr(:block, lines...)
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

    y1, y2 = generated_mlp_forward_1d(
        x,
        ps,
        Val(config.width),
        Val(config.n_hidden),
    )

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
    n_hidden = 1,
)

radiation_nllw = ManualNLLWGeneratedTest(spectral_grid; config, rng)

@info "ps structure"
@show typeof(radiation_nllw.ps)
@show keys(radiation_nllw.ps)
@show size(radiation_nllw.ps.layer_1.weight)
@show size(radiation_nllw.ps.layer_2.weight)
@show size(radiation_nllw.ps.layer_3.weight)

# Forward equivalence check against Lux.apply for one scalar input
x_test = Float32[250.0]
y_lux, _ = Lux.apply(radiation_nllw.nn, x_test, radiation_nllw.ps, radiation_nllw.st)

y1_gen, y2_gen = generated_mlp_forward_1d(
    x_test[1],
    radiation_nllw.ps,
    Val(config.width),
    Val(config.n_hidden),
)
y_gen = Float32[y1_gen, y2_gen]

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

@info "Starting Enzyme autodiff with generated scalar forward, n_hidden = 1"

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
@show maximum(abs, bmodel_ad.longwave_radiation.ps.layer_3.weight)
@show maximum(abs, bmodel_ad.longwave_radiation.ps.layer_3.bias)

@info "Script finished"