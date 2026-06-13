using Lux
using Enzyme
using Random

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

tree_l2sum(x::Number) = abs2(x)
tree_l2sum(x::AbstractArray) = sum(abs2, x)
tree_l2sum(x::NamedTuple) = sum(tree_l2sum, values(x))
tree_l2sum(x::Tuple) = sum(tree_l2sum, x)
tree_l2norm(x) = sqrt(tree_l2sum(x))

# ------------------------------------------------------------
# Build Lux model
# ------------------------------------------------------------

@info "Building Lux model"

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
st = Lux.testmode(st)

x = Float32[250.0]

@info "ps structure"
@show typeof(ps)
@show keys(ps)
@show typeof(st)

# ------------------------------------------------------------
# Loss using Lux.apply
# ------------------------------------------------------------

function lux_apply_loss(ps, x, nn, st)
    y, _ = Lux.apply(nn, x, ps, st)
    return sum(y)
end

# Same but x first, just in case
function lux_apply_loss_xfirst(x, ps, nn, st)
    y, _ = Lux.apply(nn, x, ps, st)
    return sum(y)
end

# ------------------------------------------------------------
# Forward check
# ------------------------------------------------------------

@info "Forward check"

y, _ = Lux.apply(nn, x, ps, st)
@show y
@show lux_apply_loss(ps, x, nn, st)

# ------------------------------------------------------------
# Enzyme Reverse test wrt ps
# ------------------------------------------------------------

@info "Preparing Enzyme objects"

dps = Enzyme.make_zero(ps)

@show tree_l2norm(dps)

@info "Starting Enzyme autodiff: lux_apply_loss(ps, x, nn, st)"

try
    @time Enzyme.autodiff(
        Enzyme.Reverse,
        lux_apply_loss,
        Enzyme.Active,
        Enzyme.Duplicated(ps, dps),
        Enzyme.Const(x),
        Enzyme.Const(nn),
        Enzyme.Const(st),
    )

    @info "AD finished successfully"
    @show tree_l2norm(dps)
    @show maximum(abs, dps.layer_1.weight)
    @show maximum(abs, dps.layer_2.weight)
    @show maximum(abs, dps.layer_3.weight)
    @show maximum(abs, dps.layer_4.weight)

catch err
    @error "AD failed in lux_apply_loss(ps, x, nn, st)" exception = (err, catch_backtrace())
end

# ------------------------------------------------------------
# Second variant: x first
# ------------------------------------------------------------

@info "Second Enzyme test: lux_apply_loss_xfirst(x, ps, nn, st)"

dps2 = Enzyme.make_zero(ps)

try
    @time Enzyme.autodiff(
        Enzyme.Reverse,
        lux_apply_loss_xfirst,
        Enzyme.Active,
        Enzyme.Const(x),
        Enzyme.Duplicated(ps, dps2),
        Enzyme.Const(nn),
        Enzyme.Const(st),
    )

    @info "Second AD finished successfully"
    @show tree_l2norm(dps2)
    @show maximum(abs, dps2.layer_1.weight)
    @show maximum(abs, dps2.layer_2.weight)
    @show maximum(abs, dps2.layer_3.weight)
    @show maximum(abs, dps2.layer_4.weight)

catch err
    @error "Second AD failed in lux_apply_loss_xfirst(x, ps, nn, st)" exception = (err, catch_backtrace())
end

@info "Script finished"