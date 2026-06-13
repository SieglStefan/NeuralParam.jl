### NeuralABRLW parameterization configuration code



# Configuration struct
struct NeuralABRLWConfig
    name::String                    # name of parameterization object, used in saving and loading        
    
    n_in::Int                       # nn input dimension
    n_out::Int                      # nn output dimension
    nn_config::AbstractNNConfig     # nn configuration (n_hidden, width, activation)

    zscore::ZScoreStats             # normalization parameter
end


# Convencience constructor
function NeuralABRLWConfig(
    spectral_grid::SpeedyWeather.SpectralGrid,
    nn_config::AbstractNNConfig;
    name::String = "default_name" 
)


    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers


    # XXX Calculate NN input dimension 
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
    file = "abrlw_L$(nlayers).jld2"
    zscore = ZScoreStats(file)


     # Return parameterization object
    return NeuralABRLWConfig(
        name,
        n_in,
        n_out,
        nn_config,
        zscore
    )
end
