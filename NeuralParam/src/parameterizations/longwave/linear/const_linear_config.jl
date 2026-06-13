### ConstLinearLW parameterization configuration code



# Configuration struct
struct ConstLinearLWConfig
    name::String                    # name of parameterization object, used in saving and loading        

    sc_a::Vector{Float32}           # scaling factors of a, used for normalized gradients     
    sc_b::Vector{Float32}           # -//-  
end


# Convenience constructor populating the scaling vectors
function ConstLinearLWConfig(
    spectral_grid::SpeedyWeather.SpectralGrid;
    name::String = "default_name"
)

    # Extract number of vertical layers and create scaling vectors
    nlayers = spectral_grid.nlayers

    sc_a = fill(5f-8, nlayers)
    sc_b = fill(5f-6, nlayers)

    # Return parameterization object
    return ConstLinearLWConfig(name, sc_a, sc_b)
end