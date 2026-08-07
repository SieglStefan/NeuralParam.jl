### Metrics utilities
###
### Different metrics and norms for evaluating model performance, training loss and gradients



# Root mean squared error
rmse(x, y) = sqrt(sum(abs2, x .- y) / length(x))
# Bias
bias(x, y) = sum(x .- y) / length(x)

# Weighted metrics
wmean(x, w) = sum(w .* x) / length(x)
wrmse(x, y, w) = sqrt(wmean(abs2.(x .- y), w))
wbias(x, y, w) = wmean(x .- y, w)


# Correlation
correlation(x, y) = cor(vec(x), vec(y))
# Maximal absolute difference
maxdiff(x, y) = maximum(abs.(x .- y))



# Extracts area weights of a grid from field
function area_weights(spectral_grid::SpectralGrid)
    
    # Extract grid and angles
    grid = spectral_grid.grid
    Ω = get_solid_angles(grid)          # one solid angle per ring
    rings = eachring(grid)              # point indices of every ring

    # Prepare weights array (one entry per grid point)
    w = zeros(Float32, spectral_grid.npoints)

    # Populate weights array with solid angles
    for (j, ring) in enumerate(rings)
        @views w[ring] .= Ω[j]
    end

    # Normalize weights so that mean weight = 1
    return w .* (length(w) / sum(w))
end