### IO utilities
###
### Helper functions for saving and loading data and models



# Save object, possible objects:
# - AbstractNeuralLW
# - Scaling
# - ZScoreStats
function save(object; path="", file="")

    mkpath(path)
    filepath = joinpath(path, file)

    # Save object
    JLD2.jldsave(filepath; object = to_cpu(object))

    @info "Object stored at $(filepath)!"

    return filepath
end


# Load object, possible objects:
# - AbstractNeuralLW
# - Scaling
# - ZScoreStats
function load(; path="", file="")

    filepath = joinpath(path, file)

    # Load object
    object = JLD2.load(filepath, "object")

    @info "Object loaded from $(filepath)!"

    return object
end



# Load a statistics file (zscore or scaling)
load_stats(file) = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "stats", file))



# Intialize .csv file for logging training data and create a meta data overview
function csv_init(meta; path="", file="")

    mkpath(path)
    filepath = joinpath(path, file)

    open(filepath, "w") do io
        for (k,v) in meta
            # Write meta data
            println(io, "# ", k, " = ", v)
        end
        # Write table header
        println(io, "ic,traj,epoch,loss,eta,pnorm,gnorm")
    end

    @info ".csv file created and initialized at $(filepath)!"

    return filepath
end


# Write a row of training data to .csv
function csv_row!(ic, traj, epoch, loss, eta, pnorm, gnorm; path="", file="")

    filepath = joinpath(path, file)

    open(filepath, "a") do io
        println(io, join((ic, traj, epoch, loss, eta, pnorm, gnorm), ","))
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
function build_meta(scheme, run_config)
    return merge(
        meta_scheme(scheme),
        Dict(
            "created"  => string(now()),
            "t_spinup" => run_config.t_spinup,
            "n_ic"     => run_config.n_ic,
            "n_traj"   => run_config.n_traj,
            "n_epochs" => run_config.n_epochs,
            "n_steps"  => run_config.n_steps,
            "n_gap"    => run_config.n_gap,
        )
    )
end