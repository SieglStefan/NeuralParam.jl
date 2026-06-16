### IO utilities
###
### Helper functions for saving and loading



#Save any longwave parameterization
function save_longwave(; path::String, radiation)
    
    # Create folder, create file path and save data
    mkpath(path)
    filepath = joinpath(path, ".jld2")

    JLD2.jldsave(filepath; radiation = radiation)

    return filepath
end


# Load any longwave parameterization
function load_longwave(; path::String, name::String)
    
    # Create file path and load and extract data
    filepath = joinpath(path, name * ".jld2")
    
    return JLD2.load(filepath, "radiation")
end



