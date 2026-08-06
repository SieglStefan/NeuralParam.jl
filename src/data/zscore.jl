### Zscore functions
###
### Structs and functions for doing zscore transformations



# Calculate the z-score transformation of x
@inline zscore(x, μ, σ) = (x .- μ) ./ σ

# Calculate the inverse z-score transformation of z
@inline inv_zscore(z, μ, σ) = z .* σ .+ μ



### Statistics helpers, used to build the stats file (see output_stats and input_stats)

# Mean and std over all entries
#   - non-finite entries are dropped: sst is NaN over land, lst over ocean
mean_std(x) = (f = filter(isfinite, x); (; mean = Float32[mean(f)], std = Float32[std(f)]))

# Mean and std per vertical layer (averaging over all other dimensions)
mean_std_layers(x) = (;
    mean = Float32[mean(filter(isfinite, selectdim(x, 2, k))) for k in axes(x, 2)],
    std  = Float32[std(filter(isfinite, selectdim(x, 2, k)))  for k in axes(x, 2)],
)


# Fit y = a + b*x for one column
function fit_linear(x::AbstractVector, y::AbstractVector)
    x_bar = mean(x)
    y_bar = mean(y)

    Sxx = sum(abs2, x .- x_bar)
    b = Sxx > 0 ? sum((x .- x_bar) .* (y .- y_bar)) / Sxx : 0f0

    return Float32(y_bar - b*x_bar), Float32(b)     # intercept (a), slope (b)
end

# Fit y = a + b*x1 + c*x2 for one column
function fit_linear(x1::AbstractVector, x2::AbstractVector, y::AbstractVector)
    A = hcat(ones(Float32, length(y)), Float32.(x1), Float32.(x2))
    c = A \ Float32.(y)
    return c[1], c[2], c[3]                         # a, b, c
end



# Struct holding zscore parameters
struct ZScoreStats{VI,VO}
    input_mean::VI          # input means
    input_std::VI           # input stds

    output_mean::VO         # output means
    output_std::VO          # output stds

    zscore_name::String     # name of the stats file
end


# Convenience constructor loading pre-calculated stats
function ZScoreStats(zscore_name::String, inputs, output_form, nlayers)

    # Load zscore stats
    data = load_zscore(zscore_name)

    # Reject stats generated at a different vertical resolution
    if hasproperty(data, :nlayers) && data.nlayers != nlayers
        error("Stats in $zscore_name were generated for nlayers=$(data.nlayers), but the scheme expects $nlayers.")
    end

    # Assemble input stats data.inputs in the order of the scheme's input spec
    input_mean, input_std = collect_stats(data.inputs, keys(inputs), zscore_name, "input")

    # Assemble output stats in the order decode expects
    group = getproperty(data, output_group(output_form))
    output_mean, output_std = collect_stats(group, output_keys(output_form), zscore_name, "output")


    return ZScoreStats(input_mean, input_std, output_mean, output_std, zscore_name)
end



# Load zscore stats from stats_dir
load_zscore(name) = load(; dir=stats_dir(name), file="stats.jld2")


# Concatenate the mean/std of input or output in the order of names_io
function collect_stats(data_io, names_io, name, what)

    # Prepare container
    mean = Float32[]
    std  = Float32[]

    # Collect mean/std for each var in names
    for var in names_io

        # Throw error if var is not represented in the stats file
        if !haskey(data_io, var)
            error("Stats in $name have no $what entry :$var. Available: $(keys(data_io)).")
        end

        # Collect
        append!(mean, Float32.(data_io[var].mean))
        append!(std,  Float32.(data_io[var].std))
    end

    return mean, std
end







