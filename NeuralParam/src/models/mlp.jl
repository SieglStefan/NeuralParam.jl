### Multi Layer Perceptron architecture



# MLP configuration struct
@kwdef struct MLPConfig <: AbstractNNConfig
    n_hidden::Int = 2       # number of hidden layers
    width::Int = 16         # width of neural network
    
    act = tanh              # XXX activation function
end


# Function for setting up a MLP
function setup_nn(
    nn_config::MLPConfig,
    n_in::Int,
    n_out::Int,
    rng = Random.default_rng(),
)

    # Extract nn architecture parameters
    (; n_hidden, width, act) = nn_config


    # Build nn
    hidden = [Lux.Dense(width => width, act) for _ in 1:n_hidden]

    nn = Lux.Chain(
        Lux.Dense(n_in => width, act),
        hidden...,
        Lux.Dense(width => n_out),
    )


    # Setup NN
    ps, st = Lux.setup(rng, nn)
    st = Lux.testmode(st)

    return nn, ps, st
end

