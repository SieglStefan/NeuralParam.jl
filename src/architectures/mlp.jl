### Multi Layer Perceptron architecture
###
### Defining struct and helper setup function



# MLP configuration struct
@kwdef struct MLPConfig{A} <: AbstractArchConfig
    n_hidden::Int = 2           # number of hidden layers
    width::Int = 16             # width of neural network
    
    act::A = Lux.leakyrelu          # activation function
end


# Setting up MLP architecture
function setup_arch(
    arch_config::MLPConfig,
    n_in::Int,
    n_out::Int,
    rng = Random.default_rng(),
)

    # Extract nn architecture parameters
    (; n_hidden, width, act) = arch_config


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

