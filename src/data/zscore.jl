### Zscore functions
###
### Structs and functions for doing zscore transformations



# Calculate the z-score transformation of x
@inline zscore(x, μ, σ) = (x .- μ) ./ σ

# Calculate the inverse z-score transformation of z
@inline inv_zscore(z, μ, σ) = z .* σ .+ μ



# Struct holding zscore parameters
struct ZScoreStats{VI,VO,C}
    input_mean::VI          # input means
    input_std::VI           # input stds

    output_mean::VO         # output means
    output_std::VO          # output stds

    center::C               # predictor center of the output form

    zscore_name::String     # name of the stats file
end


# Convenience constructor loading pre-calculated stats
function ZScoreStats(zscore_name::String, input_spec, output_form, nlayers)

    # Load zscore stats
    data = load_zscore(zscore_name)

    # Reject stats generated at a different vertical resolution
    if hasproperty(data, :nlayers) && data.nlayers != nlayers
        error("Stats in $zscore_name were generated for nlayers=$(data.nlayers), but the scheme expects $nlayers.")
    end

    # Assemble input stats data.inputs in the order of the scheme's input spec
    input_mean, input_std = collect_stats(data.inputs, keys(input_spec), zscore_name, "input")

    # Assemble output stats in the order decode expects
    group = output_group(output_form)
    group_data = getproperty(data, group)
    output_mean, output_std = collect_stats(group_data, output_keys(output_form), zscore_name, "output")

    # Assemle predictor center for the output form
    center = output_center(output_form, group_data, nlayers)

    return ZScoreStats(input_mean, input_std, output_mean, output_std, center, zscore_name)
end



# Load zscore stats from stats_dir
load_zscore(name) = load(; dir=stats_dir(name), file="stats.jld2")



# Concatenate the mean/std of input or output in the order of names_io
function collect_stats(data_io, names_io, zscore_name, io_type)

    # Prepare container
    mean = Float32[]
    std  = Float32[]

    # Collect mean/std for each var in names
    for var in names_io

        # Throw error if var is not represented in the stats file
        if !haskey(data_io, var)
            error("Stats in $zscore_name have no $io_type entry :$var. Available: $(keys(data_io)).")
        end

        # Collect
        append!(mean, Float32.(data_io[var].mean))
        append!(std,  Float32.(data_io[var].std))
    end

    return mean, std
end