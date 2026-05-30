### Neural network utilities for handling neural networks



# Calculate the z-score transformation of x
zscore(x, μ, σ) = (x .- μ) ./ σ

# Calculate the inverse z-score transformation of z
inverse_zscore(z, μ, σ) = z .* σ .+ μ



# Function for setting up a NN for a NeuralABRLongwave parameterization
function setup_nn(
    config,
    rng = Random.default_rng(),
)

    # Build NN layers
    nn =  build_nn(config)

    # Setup NN
    ps, st = Lux.setup(rng, nn)
    st = Lux.testmode(st)

    return nn, ps, st
end


# Build a fully-connected Lux neural network
function build_nn(config)

    # Extract NN parameters
    width = config.width
    n_hidden = config.n_hidden

    # Choose activation function
    if config.activation == :tanh
        act = tanh
    else
        @warn "Activation function not defined! tanh is used"
        act = tanh
    end

    # Create hidden layers and NN
    hidden = [Lux.Dense(width => width, act) for _ in 1:n_hidden]

    nn = Lux.Chain(
        Lux.Dense(config.n_in => width, act),
        hidden...,
        Lux.Dense(width => config.n_out),
    )

    return nn
end



# Normalizes (zscore) NN input 
function normalize_nn_input(radiation, x)
    return zscore(
        Float32.(x),
        radiation.config.input_mean,
        radiation.config.input_std
    )
end


# Unscales (inverse zscore) NN output for linear LW
function unscale_nn_output(para::AbstractLinearLongwave, y)
    
    a = para.config.sc_a * y[1:2:end]
    b = para.config.sc_b * y[2:2:end] 

    return a, b
end

# Unscales (inverse zscore) NN output for ABR
function unscale_nn_output(para::AbstractNeuralABRLongwave, y)
    return inverse_zscore(
        Float32.(y), 
        para.config.output_mean, 
        para.config.output_std
    )
end