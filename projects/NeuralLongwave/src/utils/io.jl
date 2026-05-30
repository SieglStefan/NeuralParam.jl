### IO utilities
###
### Helper functions for saving and loading NeuralLinearLongwave
### parameterizations using JLD2 and zscore statistics.



# Save a neural longwave parameterization using JLD2
function save_neural_longwave(; path::String, radiation::AbstractNeuralLinearLongwave)
    
    # Create folder, create file path and save data
    mkpath(path)
    filepath = joinpath(path, radiation.config.name * ".jld2")

    JLD2.jldsave(
        filepath;
        config = radiation.config,
        ps = radiation.ps,
        st = radiation.st,
    )

    return filepath
end


# Load a neural longwave parameterization as Lux inference version
function load_neural_longwave(; path::String, name::String, SG::SpeedyWeather.SpectralGrid)
    
    # Create file path and load and extract data
    filepath = joinpath(path, name * ".jld2")
    data = JLD2.load(filepath)

    config = data["config"]
    ps = data["ps"]
    st = data["st"]

    # Rebuild matching Lux architecture
    tmp = NeuralLinearLongwave(SG; config)

    return NeuralLinearLongwave(;
        nn = tmp.nn,
        ps,
        st,
        config,
    )
end



# Function for loading pre-generated zscore statistics
function load_zscore(file)
    path = joinpath(@__DIR__, "..", "..", "data", "zscore", file)

    if !isfile(path)
        error("Z-score statistics file not found: $path")
    end

    data = JLD2.load(path)

    return data
end