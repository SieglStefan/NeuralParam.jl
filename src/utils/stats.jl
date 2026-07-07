### Statistics utilities
###
### - Defining structs and helper functions for output scaling
### - Defining structs and helper functions for zscore-statistics for input and output



# Struct holding output scaling parameters
struct Scaling{V}
    sc_a::V          
    sc_b::V       
end


# Convenience constructor loading pre-calculated stats and pushing them on arch
function Scaling(folder::String, arch = SpeedyWeather.CPU())

    data = load_stats(folder)

    return Scaling(
        on_architecture(arch, Float32.(data["sc_a"])),
        on_architecture(arch, Float32.(data["sc_b"])),
    )
end

# Conveience constructor for standard scaling
Scaling(nlayers::Int) = Scaling(fill(5f-8, nlayers), fill(5f-6, nlayers))



# Calculate the z-score transformation of x
@inline zscore(x, μ, σ) = (x .- μ) ./ σ

# Calculate the inverse z-score transformation of z
@inline inv_zscore(z, μ, σ) = z .* σ .+ μ


# Struct holding zscore parameters
struct ZScoreStats{V}
    input_mean::V
    input_std::V

    output_mean::V
    output_std::V
end


# Convenience constructor loading pre-calculated stats and push them on arch
function ZScoreStats(folder::String, arch = SpeedyWeather.CPU())
    
    data = load_stats(folder)

    return ZScoreStats(
        on_architecture(arch,   Float32.(data["input_mean"])),
        on_architecture(arch,   Float32.(data["input_std"])),
        on_architecture(arch,   Float32.(data["output_mean"])),
        on_architecture(arch,   Float32.(data["output_std"])),
    )
end
