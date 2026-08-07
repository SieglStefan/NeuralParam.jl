### IO utilities
###
### Helper functions for saving and loading objects, data and information



# Function for saving an object to a .jld2 file
function save(object; dir, file)
    
    # Create path and put together filepath
    mkpath(dir)
    filepath = joinpath(dir, file)

    # Save object
    JLD2.jldsave(filepath; object)

    return filepath
end

# Function for loading an object from a .jld2 file
function load(; dir, file)

    # Load object
    filepath = joinpath(dir, file)
    object = JLD2.load(filepath, "object")

    return object
end



# Define root path of the project
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# Define paths for difference data folders
reference_dir(parts...) = joinpath(ROOT, "data", "reference", parts...)     # reference data sets for training and testing
stats_dir(parts...)    = joinpath(ROOT, "data", "stats", parts...)        # statistics used for zscore
scheme_dir(parts...)    = joinpath(ROOT, "results", "schemes", parts...)    # calibrated and trained schemes
rollout_dir(parts...)   = joinpath(ROOT, "results", "rollouts", parts...)   # generated rollouts for evaluation and testing


# Utility functions for collecting and loading multiple schemes and rollouts
collect_schemes(names)  = (; (Symbol(n) => load(; dir=scheme_dir(n),  file="scheme.jld2")  for n in names)...)
collect_rollouts(names) = (; (Symbol(n) => load(; dir=rollout_dir(n), file="rollout.jld2") for n in names)...)


# Utility functions for resolving schemes
resolve_scheme(name::String) = load(; dir=scheme_dir(name), file="scheme.jld2")
resolve_scheme(scheme)       = scheme

# Utility functions for scheme names
scheme_name(name::AbstractString) = name
scheme_name(::Nothing)            = "none"
scheme_name(scheme)               = string(nameof(typeof(scheme)))



# Utility functions for converting objects to TOML-compatible types for info.toml
_toml(x)                 = x
_toml(x::AbstractFloat)  = Float64(x)                                             
_toml(x::Symbol)         = string(x)
_toml(x::AbstractDict)   = Dict(string(k) => _toml(v) for (k, v) in x)
_toml(x::NamedTuple)     = Dict(string(k) => _toml(v) for (k, v) in pairs(x))
_toml(x::AbstractVector) = [_toml(v) for v in x]

# Writes a info.toml file with the given keyword arguments
function write_info(; dir="", file="", kwargs...)
    
    # Create info dictonary out of keyword arguments
    info = Dict(string(k) => _toml(v) for (k, v) in kwargs)

    # Create path and put together filepath
    mkpath(dir)
    filepath = joinpath(dir, file)

    # Write info to .toml file
    open(filepath, "w") do io
        TOML.print(io, info; sorted = true)   # sorted = stable, diff-friendly
    end

    # Print information
    @info "Info file written to $(filepath)!"

    return filepath
end



# Function for preparing an output directory for writing results, throws error if folder already exists
function prepare_out_dir(base_dir, name)
    out_dir = joinpath(base_dir, name)

    # Check if folder already exists and throw error
    if isdir(out_dir)
        error("Folder already exists ($out_dir): Task canceled! (not overwritten).")
    end

    return mkpath(out_dir)
end

# Function for preparing an output directory for writing results, deletes folder if it already exists
function fresh_out_dir(base_dir, name)
    out_dir = joinpath(base_dir, name)

    # Check if folder already exists and delete it
    if  isdir(out_dir)
        rm(out_dir; recursive = true) 
    end
    
    return mkpath(out_dir)
end





### Lazily-read reference dataset: one full model state per simulated day
struct Reference{S}
    store::S            # open .jld2 file — states are read on demand, not held in RAM
    sim_days::Int       # last available day; valid days are 0:sim_days
end

# Index by simulated day: ref[0] is the initial state, ref[12] the state after 12 days
Base.getindex(ref::Reference, day::Integer) = ref.store["day_$(day)"]

# Open a stored reference for the duration of the callback
function with_reference(fn, name::AbstractString)
    JLD2.jldopen(joinpath(reference_dir(name), "reference.jld2"), "r") do store
        fn(Reference(store, store["sim_days"]))
    end
end


# Open a .jld2 store for incremental writing; the callback receives the open file
function save_store(fn; dir, file)
    
    # Create directory and creaet filepath
    mkpath(dir)
    filepath = joinpath(dir, file)

    # Open the file and call the callback
    JLD2.jldopen(filepath, "w") do store
        fn(store)
    end

    return filepath
end