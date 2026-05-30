### Shared config and helper functions for NeuralABRLongwave parameterizations
###
### Important:
### - Note: Loaded zscore statistics must match the NN input/output ordering.



# Convenience container for the parameters of a NeuralABRLongwave NN
struct NeuralABRLongwaveConfig
    name::String                    # name of model used for storing and loading
    
    n_in::Int                       # number of inputs of the NN
    n_out::Int                      # number of outputs of the NN

    width::Int                      # number neurons per hidden layer
    n_hidden::Int                   # number of hidden layers
    activation::Symbol              # activation function
    
    default_CO2::Float32            # default CO2 concentration set to modern times

    input_mean::Vector{Float32}     # input zscore data mean,   calculated in: NeuralLongwave/scripts/zscore_abrlw.jl
    input_std::Vector{Float32}      # input zscore data std,    calculated in: -//-

    output_mean::Vector{Float32}    # output zscore data mean,  calculated in: -//-
    output_std::Vector{Float32}     # output zscore data std,   calculated in: -//-
end


# Constructor for load zscore data and filling the mean and std arrays
function NeuralABRLongwaveConfig(
    spectral_grid::SpeedyWeather.SpectralGrid;
    name::String = "default_name",
    width::Int = 32,
    n_hidden::Int = 2,
    activation::Symbol = :tanh,
    default_CO2::Float32 = 400f0
)

    # Extract vertical layers and calculate input and calculate dimension
    trunc = spectral_grid.trunc
    nlayers = spectral_grid.nlayers


    # Calculate NN input dimension 
    # - temperature profile:        nlayers
    # - humidity profile:           nlayers
    # - surface_pressure:           1
    # - sea_surface_temperature:    1
    # - land_surface_temperature:   1
    # - land_fraction:              1
    n_in = 2*nlayers + 4

    # Calculate NN output dimension
    n_out = nlayers


    # Load zscore statistics 
    file = "abrlw_T$(trunc)_L$(nlayers).jld2"
    data = load_zscore(file)

    input_mean  = data["input_mean"]
    input_std   = data["input_std"]
    output_mean = data["output_mean"]
    output_std  = data["output_std"]


    # Populate and return Config struct
    return NeuralABRLongwaveConfig(
        name,
        n_in,
        n_out,
        width,
        n_hidden,
        activation,
        default_CO2,
        input_mean,
        input_std,
        output_mean,
        output_std
    )
end
