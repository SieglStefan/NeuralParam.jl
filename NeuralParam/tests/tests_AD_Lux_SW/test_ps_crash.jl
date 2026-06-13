using Lux
using Enzyme
using Random

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

tree_l2sum(x::Number) = abs2(x)
tree_l2sum(x::AbstractArray) = sum(abs2, x)
tree_l2sum(x::NamedTuple) = sum(tree_l2sum, values(x))
tree_l2sum(x::Tuple) = sum(tree_l2sum, x)
tree_l2norm(x) = sqrt(tree_l2sum(x))

# ------------------------------------------------------------
# Manual Lux-like forward, statically unrolled
# Corresponds to:
#
# Chain(
#     Dense(1 => width, tanh),
#     Dense(width => width, tanh),
#     Dense(width => width, tanh),
#     Dense(width => 2),
# )
#
# i.e. 3 hidden activations and one output layer.
# ------------------------------------------------------------

function manual_mlp_forward(x, ps, ::Val{2})
    p1 = ps.layer_1
    h1 = tanh.(p1.weight * x .+ p1.bias)

    p2 = ps.layer_2
    h2 = tanh.(p2.weight * h1 .+ p2.bias)

    p3 = ps.layer_3
    h3 = tanh.(p3.weight * h2 .+ p3.bias)

    p4 = ps.layer_4
    y = p4.weight * h3 .+ p4.bias

    return y
end

# Scalar loss for Enzyme Reverse.
# ps is the differentiated object.
# x is kept constant.
function mlp_loss(ps, x)
    y = manual_mlp_forward(x, ps, Val(2))
    return sum(y)
end

# Same loss but with arguments reversed, in case Enzyme behaves better.
function mlp_loss_xfirst(x, ps)
    y = manual_mlp_forward(x, ps, Val(2))
    return sum(y)
end

# ------------------------------------------------------------
# Build Lux model only to get Lux-like ps structure
# ------------------------------------------------------------

@info "Building Lux model and parameters"

rng = Random.default_rng()
Random.seed!(rng, 1234)

width = 4

nn = Lux.Chain(
    Lux.Dense(1 => width, tanh),
    Lux.Dense(width => width, tanh),
    Lux.Dense(width => width, tanh),
    Lux.Dense(width => 2),
)

ps, st = Lux.setup(rng, nn)

# Use Float32 input, same spirit as SpeedyWeather
x = Float32[250.0]

@info "Lux ps structure"
@show typeof(ps)
@show keys(ps)
@show typeof(ps.layer_1)
@show size(ps.layer_1.weight)
@show size(ps.layer_1.bias)

# ------------------------------------------------------------
# Forward sanity check
# ------------------------------------------------------------

@info "Forward sanity check"

y_manual = manual_mlp_forward(x, ps, Val(2))
y_lux, _ = Lux.apply(nn, x, ps, Lux.testmode(st))

@show y_manual
@show y_lux
@show maximum(abs, y_manual .- y_lux)

# ------------------------------------------------------------
# Enzyme Reverse test: differentiate wrt ps
# ------------------------------------------------------------

@info "Preparing Enzyme objects"

dps = Enzyme.make_zero(ps)

@show tree_l2norm(dps)

@info "Starting Enzyme autodiff: mlp_loss(ps, x)"

try
    @time Enzyme.autodiff(
        Enzyme.Reverse,
        mlp_loss,
        Enzyme.Active,
        Enzyme.Duplicated(ps, dps),
        Enzyme.Const(x),
    )

    @info "AD finished successfully"
    @show tree_l2norm(dps)
    @show maximum(abs, dps.layer_1.weight)
    @show maximum(abs, dps.layer_2.weight)
    @show maximum(abs, dps.layer_3.weight)
    @show maximum(abs, dps.layer_4.weight)

catch err
    @error "AD failed in mlp_loss(ps, x)" exception = (err, catch_backtrace())
end

# ------------------------------------------------------------
# Optional second variant: x first, ps second
# ------------------------------------------------------------

@info "Second Enzyme test: mlp_loss_xfirst(x, ps)"

dps2 = Enzyme.make_zero(ps)

try
    @time Enzyme.autodiff(
        Enzyme.Reverse,
        mlp_loss_xfirst,
        Enzyme.Active,
        Enzyme.Const(x),
        Enzyme.Duplicated(ps, dps2),
    )

    @info "Second AD finished successfully"
    @show tree_l2norm(dps2)
    @show maximum(abs, dps2.layer_1.weight)
    @show maximum(abs, dps2.layer_2.weight)
    @show maximum(abs, dps2.layer_3.weight)
    @show maximum(abs, dps2.layer_4.weight)

catch err
    @error "Second AD failed in mlp_loss_xfirst(x, ps)" exception = (err, catch_backtrace())
end

@info "Script finished"