### IO utilities
###
### Helper functions for saving and loading objects and data



# Function for saving an object to a .jld2 file
function save(object; path, file)
    
    # Create path and put together filepath
    mkpath(path)
    filepath = joinpath(path, file)

    # Save object
    JLD2.jldsave(filepath; object)

    return filepath
end

# Function for loading an object from a .jld2 file
function load(; path, file)

    # Load object
    filepath = joinpath(path, file)
    object = JLD2.load(filepath, "object")

    return object
end



# Intialize .csv file for logging training data and create a info.toml file for meta data
function csv_init(metric_keys; path="", file="")

    # Create path and put together filepath
    mkpath(path)
    filepath = joinpath(path, file)

    # Write header
    open(filepath, "w") do io
        println(io, "ic,traj,epoch,n_steps,loss,eta,pnorm,gnorm," * join(metric_keys, ","))
    end

    # Print information
    @info ".csv file created and initialized at $(filepath)!"

    return filepath
end


# Write a row of training data to .csv
function csv_row!(ic, traj, epoch, n_steps, loss, eta, pnorm, gnorm, metrics; path="", file="")

    # Put together filepath
    filepath = joinpath(path, file)

    # Write a data row
    open(filepath, "a") do io
        println(io, join((ic, traj, epoch, n_steps, loss, eta, pnorm, gnorm, values(metrics)...), ","))
    end

    return nothing
end


# Read .csv data for plotting
function csv_read(; path="", file="")

    # Put together filepath
    filepath = joinpath(path, file)

    # Read .csv data
    return CSV.read(filepath, DataFrame; comment="#")
end



















# Make values TOML-serializable
_toml(x)                = x
_toml(x::AbstractFloat) = Float64(x)
_toml(x::Symbol)        = string(x)
_toml(x::AbstractDict)  = Dict(string(k) => _toml(v) for (k, v) in x)
_toml(x::NamedTuple)     = Dict(string(k) => _toml(v) for (k, v) in pairs(x))
_toml(x::AbstractVector) = [_toml(v) for v in x]

# Write stats meta data into .toml file
function write_info(; path="", file="", kwargs...)
    meta = Dict(string(k) => _toml(v) for (k, v) in kwargs)
    mkpath(path)
    filepath = joinpath(path, file)
    open(filepath, "w") do io
        TOML.print(io, meta; sorted = true)   # sorted = stable, diff-friendly
    end
    @info "Info file written to $(filepath)!"
    return filepath
end




# XXX
function prepare_out_dir(folderpath, folder_name)
    out_dir = joinpath(folderpath, folder_name)

    if isdir(out_dir)
        error("Folder already exists ($out_dir): Task canceled! (not overwritten).")
    end
    return mkpath(out_dir)
end


# XXX
function fresh_out_dir(folderpath, folder_name)
    out_dir = joinpath(folderpath, folder_name)

    if  isdir(out_dir)
        rm(out_dir; recursive = true) 
    end
    
    return mkpath(out_dir)
end