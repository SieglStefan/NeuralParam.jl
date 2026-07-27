### IO utilities
###
### Helper functions for saving and loading objects and data

###
###
###




const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

reference_dir(name) = joinpath(ROOT, "data", "reference", name)
stats_dir(name)     = joinpath(ROOT, "data", "stats", name)
model_dir(name)     = joinpath(ROOT, "results", "models", name)
rollout_dir(name)   = joinpath(ROOT, "results", "rollouts", name)

collect_schemes(names)  = (; (Symbol(n) => load(; path=model_dir(n),   file="scheme.jld2")  for n in names)...)
collect_rollouts(names) = (; (Symbol(n) => load(; path=rollout_dir(n), file="rollout.jld2") for n in names)...)
load_stats(name)        = load(; path=stats_dir(name), file="stats.jld2")





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