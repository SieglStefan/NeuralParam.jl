### Utility functions for statistics used in parameterization
###
### - Mainly for zscore normalisation input_/output_mean and std
### - Also includes functions for scaling parameters sc_a and sc_b



# Calculate the z-score transformation of x
@inline zscore(x, μ, σ) = (x .- μ) ./ σ

# Calculate the inverse z-score transformation of z
@inline inv_zscore(z, μ, σ) = z .* σ .+ μ



# Struct holding zscore parameters previously calculated
struct ZScoreStats
    input_mean::Vector{Float32}
    input_std::Vector{Float32}

    output_mean::Vector{Float32}
    output_std::Vector{Float32}
end


# Convenience constructor loading pre-calculated stats
function ZScoreStats(file::String)
    data = load_zscore(file)

    return ZScoreStats(
        data["input_mean"],
        data["input_std"],
        data["output_mean"],
        data["output_std"]
    )
end


# Function for loading pre-calculated zscore statistics
function load_zscore(file)
    path = joinpath(@__DIR__, "..", "..", "data", "zscore", file)

    if !isfile(path)
        error("Z-score statistics file not found: $path")
    end

    data = JLD2.load(path)

    return data
end



# Function for loading pre-calculated a and b output scaling factors
function load_output_scaling(file)
    path = joinpath(@__DIR__, "..", "..", "data", "scaling", file)

    if !isfile(path)
        error("Output scaling file not found: $path")
    end

    data = JLD2.load(path)

    return data["sc_a"], data["sc_b"]
end



