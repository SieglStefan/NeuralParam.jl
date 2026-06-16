### NeuralLinearLW parameterization configuration code



# Configuration struct
struct NeuralLinearLWConfig
    name::String                    # name of parameterization object, used in saving and loading        
    
    n_in::Int                       # nn input dimension
    n_out::Int                      # nn output dimension
    nn_config::AbstractNNConfig     # nn configuration (n_hidden, width, activation)

    zscore::ZScoreStats             # normalization parameter only holding input statistics

    sc_a::Vector{Float32}           # scaling factor of a (result of ConstLinearLW calibration)
    sc_b::Vector{Float32}           # scaling factor of b (-//-)
end


# Convenience constructor extracting input/output dimension and loading zscore statistics
function NeuralLinearLWConfig(
    spectral_grid::SpeedyWeather.SpectralGrid,
    nn_config::AbstractNNConfig;
    name::String = "default_name" 
)

    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers

    
    # Calculate NN input dimension
    # - temperature profile: nlayers    
    n_in = nlayers

    # Calculate NN output dimension
    # - a: nlayers
    # - b: nlayers
    n_out = 2*nlayers


    # Load zscore statistics and output scaling parameters
    file = "zscore_abrlw_L$(nlayers).jld2"
    zscore = ZScoreStats(file)
    #sc_a, sc_b = load_scaling(file, nlayers)

    sc_a = fill(5f-8, nlayers)
    sc_b = fill(5f-6, nlayers)

    # Return parameterization object
    return NeuralLinearLWConfig(
        name,
        n_in,
        n_out,
        nn_config,
        zscore,
        sc_a,
        sc_b
    )
end