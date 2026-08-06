### General utility functions
###
### Helper functions used across the codebase



# Extracts vertical layer k from a series of fields
extract_layer(layer, f) = [i[:, layer] for i in f]



# Calculate the number of steps from number of days
steps_from_days(days, Δt_sec) = round(Int, days * 86400 / Δt_sec)

# Calculate the number of days from number of steps
days_from_steps(n_steps, Δt_sec) = n_steps * Δt_sec / 86400



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