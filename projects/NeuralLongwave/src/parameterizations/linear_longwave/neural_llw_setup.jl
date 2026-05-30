### Shared config and helper functions for NeuralLinearLongwave parameterizations
###
### Important:
### - XXX Both NeuralLinearLongwave and NeuralLinearLongwaveAD use the same Lux architecture.
### - XXX NeuralLinearLongwave uses Lux.apply() in parameterization!().
### - XXX NeuralLinearLongwaveAD uses a generated scalar forward pass in parameterization!().
### - Note: Loaded zscore statistics must match the NN input/output ordering.



# Convenience container for the parameters of a NeuralLinearLongwave NN
struct NeuralLinearLongwaveConfig
    name::String                    # name of model used for storing and loading
    
    n_in::Int                       # number of inputs of the NN
    n_out::Int                      # number of outputs of the NN

    width::Int                      # number neurons per hidden layer
    n_hidden::Int                   # number of hidden layers
    activation::Symbol              # activation function
    
    input_mean::Vector{Float32}     # input zscore data mean, calculated in: NeuralLongwave/scripts/zscore_abrlw.jl
    input_std::Vector{Float32}      # input zscore data std,  calculated in: -//-

    sc_a::Float32                   # scaling factor for a-output (calculated from calibration runs in experiments/XXX)
    sc_b::Float32                   # scaling factor for b-output (-//-)
end



# Constructor for load zscore data and filling the mean and std arrays
function NeuralLinearLongwaveConfig(
    spectral_grid::SpeedyWeather.SpectralGrid;
    name::String = "default_name",
    width::Int = 32,
    n_hidden::Int = 2,
    activation::Symbol = :tanh,
    sc_a = 5.5f-8,
    sc_b::Float32 = 4.6f-7
)

    # Extract grid parameters
    trunc = spectral_grid.trunc
    nlayers = spectral_grid.nlayers


    # Calculate NN input dimension
    # - temperature profile: nlayers    
    n_in = nlayers

    # Calculate NN output dimension
    # - a: nlayers
    # - b: nlayers
    n_out = 2*nlayers


    # Load zscore statistics 
    file = "abrlw_T$(trunc)_L$(nlayers).jld2"
    data = load_zscore(file)

    input_mean  = data["input_mean"]
    input_std   = data["input_std"]


    # Populate and return Config struct
    return NeuralLinearLongwaveConfig(
        name,
        n_in,
        n_out,
        width,
        n_hidden,
        activation,
        input_mean,
        input_std,
        sc_a,
        sc_b
    )
end