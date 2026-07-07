### IO utilities
###
### Helper functions for saving and loading data and models



# Saves a parameterization scheme as .jld2
function save_scheme(scheme; path="", file="")

    mkpath(path)
    filepath = joinpath(path, file)

    # Save scheme
    JLD2.jldsave(filepath; scheme = to_cpu(scheme))

    @info "Scheme stored at $(filepath)!"

    return filepath
end


# Loads a .jld2 parameterization scheme
function load_scheme(; path="", file="")

    filepath = joinpath(path, file)

    # Load scheme
    scheme = JLD2.load(filepath, "scheme")

    @info "Scheme loaded from $(filepath)!"

    return scheme
end



# Load a statistics file (zscore or scaling)
load_stats(folder; file="stats.jld2") = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "stats", folder, file))



# Intialize .csv file for logging training data and create a meta data overview
function csv_init(meta, metric_keys; path="", file="")

    mkpath(path)
    filepath = joinpath(path, file)

    open(filepath, "w") do io
        for (k,v) in meta
            # Write meta data
            println(io, "# ", k, " = ", v)
        end
        # Write table header
        println(io, "ic,traj,epoch,n_steps,loss,eta,pnorm,gnorm," * join(metric_keys, ","))
    end

    @info ".csv file created and initialized at $(filepath)!"

    return filepath
end


# Write a row of training data to .csv
function csv_row!(ic, traj, epoch, n_steps, loss, eta, pnorm, gnorm, metrics; path="", file="")

    filepath = joinpath(path, file)

    open(filepath, "a") do io
        println(io, join((ic, traj, epoch, n_steps, loss, eta, pnorm, gnorm, values(metrics)...), ","))
    end

    return nothing
end


# Read and print meta data of a training .csv file
function csv_info(; path="", file="")

    filepath = joinpath(path, file)
    meta = Dict{String,String}()

    # Read meta
    for line in eachline(filepath)
        startswith(line, "#") || break           
        k, v = split(strip(line[2:end]), " = "; limit=2)
        meta[strip(k)] = strip(v)
    end

    return meta
end


# Read .csv data for plotting and printing information
function csv_read(; path="", file="")

    filepath = joinpath(path, file)

    return CSV.read(filepath, DataFrame; comment="#")
end



# Define meta data for a MLP neural network architecture
arch_meta(c::MLPConfig) = Dict(
    "n_hidden"  =>  c.n_hidden, 
    "width"     =>  c.width, 
    "act"       =>  string(c.act)
)


# Define meta data for a ConstLinearLW parameterization
meta_scheme(s::ConstLinearLW) = Dict(
    "scheme"    =>  "ConstLinearLW"
)

# Define meta data for a NeuralLinearLW parameterization
meta_scheme(s::NeuralLinearLW) = merge(Dict(
    "scheme"    =>  "NeuralLinearLW",   
    "n_in"      =>  s.n_in, 
    "n_out"     =>  s.n_out), 
    arch_meta(s.arch_config)
)

# Define meta data for a NeuralABRLW parameterization
meta_scheme(s::NeuralABRLW) = merge(Dict(
    "scheme"    =>  "NeuralABRLW",       
    "n_in"      =>  s.n_in, 
    "n_out"     =>  s.n_out), 
    arch_meta(s.arch_config)
)

# Define meta data for a NeuralABRLWGlobal parameterization
meta_scheme(s::NeuralABRLWGlobal) = merge(Dict(
    "scheme"    =>  "NeuralABRLWGlobal", 
    "n_in"      =>  s.n_in, 
    "n_out"     =>  s.n_out, 
    "n_points"  =>  s.n_points), 
    arch_meta(s.arch_config)
)


# Merge the meta data together
function build_meta(scheme, target, run_config)
    target_name = isnothing(target) ? "default" : string(nameof(typeof(target)))
    return merge(
        meta_scheme(scheme),
        Dict(string(f) => getfield(run_config, f) for f in fieldnames(typeof(run_config))),  # all config fields
        Dict(
            "created"       => string(now()),
            "julia"         => string(VERSION),
            "target_scheme" => target_name,
        ),
    )
end



# Make values TOML-serializable
_toml(x)                = x
_toml(x::AbstractFloat) = Float64(x)
_toml(x::Symbol)        = string(x)
_toml(x::AbstractDict)  = Dict(string(k) => _toml(v) for (k, v) in x)

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