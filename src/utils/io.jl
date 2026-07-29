### IO utilities
###
### Helper functions for saving and loading objects and data

###
###
###




const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

reference_dir(parts...) = joinpath(ROOT, "data", "reference", parts...)
stats_dir(parts...)     = joinpath(ROOT, "data", "stats", parts...)
scheme_dir(parts...)     = joinpath(ROOT, "results", "schemes", parts...)
rollout_dir(parts...)   = joinpath(ROOT, "results", "rollouts", parts...)

collect_schemes(names)  = (; (Symbol(n) => load(; path=scheme_dir(n),   file="scheme.jld2")  for n in names)...)
collect_rollouts(names) = (; (Symbol(n) => load(; path=rollout_dir(n), file="rollout.jld2") for n in names)...)
load_stats(name)        = load(; path=stats_dir(name), file="stats.jld2")


resolve_scheme(name::String) = load(; path=scheme_dir(name), file="scheme.jld2")
resolve_scheme(scheme) = scheme

scheme_name(name::AbstractString) = name
scheme_name(::Nothing)            = "none"
scheme_name(scheme)               = string(nameof(typeof(scheme)))




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

    # Check if folder already exists and throw error
    if isdir(out_dir)
        error("Folder already exists ($out_dir): Task canceled! (not overwritten).")
    end
    return mkpath(out_dir)
end


# XXX
function fresh_out_dir(folderpath, folder_name)
    out_dir = joinpath(folderpath, folder_name)

    # Check if folder already exists and delete it
    if  isdir(out_dir)
        rm(out_dir; recursive = true) 
    end
    
    return mkpath(out_dir)
end




### XXX Lazily-read reference dataset: one full model state per simulated day
struct Reference{S}
    store::S            # open .jld2 file — states are read on demand, not held in RAM
    sim_days::Int       # last available day; valid days are 0:sim_days
end

# XXX Index by simulated day directly: ref[0] is the initial state, ref[12] the state after 12 days
Base.getindex(ref::Reference, day::Integer)  = ref.store["day_$(day)/full"]


# XXX Grid fields only — much smaller read, used for the metric comparisons
grid_state(ref::Reference, day::Integer)     = ref.store["day_$(day)/grid"]

# XXX Open a stored reference for the duration of the callback
function with_reference(fn, name::AbstractString)
    JLD2.jldopen(joinpath(reference_dir(name), "reference.jld2"), "r") do store
        fn(Reference(store, store["sim_days"]))
    end
end


# Open a .jld2 store for incremental writing; the callback receives the open file
function save_store(fn; path, file)
    mkpath(path)
    filepath = joinpath(path, file)
    JLD2.jldopen(filepath, "w") do store
        fn(store)
    end
    return filepath
end