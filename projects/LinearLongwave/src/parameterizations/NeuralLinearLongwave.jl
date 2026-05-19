### Structs and functions for the NeuralLinearLongwave Parameterization


# Convenience container for the parameters of a NeuralLinearLongwave NN
@kwdef struct NeuralLinearLongwaveConfig
    name::String = "default_name"           # name of model
    width::Int = 32                         # number neurons per hidden layer
    n_hidden::Int = 2                       # number of hidden layers
    activation::Symbol = :tanh              # activation function
    sc_a::Float32 = 1f-6                    # scaling factor for a-output
    sc_b::Float32 = 1f-3                    # scaling factor for b-output
    T_mean::Float32 = 249.83                # mean temperature for zscore, calculated in "zscore_calc_params.jl"
    T_std::Float32 =  23.09                 # temperature std for zscore, calculated in "zscore_calc_params.jl"
end





# Defines a parameterization scheme for linear longwave radiation: dTi = ai * Ti + bi, where ai and bi are outputs of a NN
@kwdef mutable struct NeuralLinearLongwave{M,P,S} <: SpeedyWeather.AbstractLongwave
    nn::M                                   # neural network (Lux)
    ps::P                                   # parameters of the NN (Lux)
    st::S                                   # state of the NN (Lux)
    config::NeuralLinearLongwaveConfig      # configuration of the NN
end


# Constructor: Creates the NN architecture regarding the spectral grid and NeuralLinearLongwave.config
function NeuralLinearLongwave(
        SG::SpeedyWeather.SpectralGrid; 
        config = NeuralLinearLongwaveConfig(),
        rng = Random.default_rng())
    
    # Extract spectal grid parameters
    n_in = SG.nlayers
    n_out = 2*n_in

    # Extract NN parameters
    width = config.width
    n_hidden = config.n_hidden
    act = config.activation


    # Create hidden layers and NN
    hidden = [Lux.Dense(width => width, act) for _ in 1:n_hidden]

    nn = Lux.Chain(
        Lux.Dense(n_in => width, act),
        hidden...,
        Lux.Dense(width => n_out),
    )

    # Setup NN
    ps, st = Lux.setup(rng, nn)

    # Returns a NeuralLinearLongwave with built NN
    return NeuralLinearLongwave(; nn, ps, st, config)
end


# Initializing function
function SpeedyWeather.initialize!(::NeuralLinearLongwave, ::AbstractModel)
    return nothing
end


# Calculate tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(ij, vars::Variables, para::NeuralLinearLongwave, model::AbstractModel)
    
    # Extract parameterization fields
    (; nn, ps, st) = para

    # Get the current temperature Tij at grid-cell ij and zscore-trafo it
    Tij = vars.grid.temperature[ij,:]
    Tij_norm = zscore(Tij, para.config.T_mean, para.config.T_std)

    # Apply the NN to Tij_norm
    y, _ = Lux.apply(nn, Tij_norm, ps, st)

    # Calculate a and b (with scaling factors, because the NN outputs ~ 1)
    a = para.config.sc_a .* y[1:2:end]
    b = para.config.sc_b .* y[2:2:end]

    # Update tendencies for each vertical grid-cell
    for k in eachindex(Tij)
        vars.tendencies.grid.temperature[ij,k] += a[k] * Tij[k] + b[k]
    end

    return nothing
end





# Function for saving a NeuralLinearLongwave model using JLD2
function save(; path::String, para::NeuralLinearLongwave)
    
    # Create folder and create file path
    mkpath(path)
    filepath = joinpath(path, para.config.name * ".jld2")
    
    # Save it
    JLD2.jldsave(
        filepath;  
        config = para.config,
        ps = para.ps,
        st = para.st,
    )

    return filepath
end


# Function for loading a NeuralLinearLongwave model
function load_neural_longwave(; path::String, name::String, SG::SpeedyWeather.SpectralGrid)
    
    # Create file path and load data
    filepath = joinpath(path, name * ".jld2")
    data = JLD2.load(filepath)

    # Extract data
    config = data["config"]
    ps = data["ps"]
    st = data["st"]

    # Create temporary NeuralLinearLongwave (NN is empty)
    tmp = NeuralLinearLongwave(SG; config=config)

    # Return initialized NLL
    return NeuralLinearLongwave(tmp.nn, ps, st, config)

end