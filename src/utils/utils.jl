### General utility functions
###
### XXX Helper functions used across



# Extract vertical layer k from a series of fields
extract_layer(layer, f) = [i[:, layer] for i in f]



# Calculate the number of steps from number of days
steps_from_days(days, Δt_sec) = round(Int, days * 86400 / Δt_sec)

# Calculate the number of days from number of steps
days_from_steps(n_steps, Δt_sec) = n_steps * Δt_sec / 86400



# Sample starting dates uniformly across the year
function sample_start_date(ic, n_ic; year=2000)
    bin = 365 / n_ic
    doy = (ic-1)*bin + rand()*bin
    return DateTime(year, 1, 1) + Day(floor(Int, doy))
end



# Calulates extrema of a traj of layer for common colorbar range
function target_colorrange(traj; layer)
    vals = Float32[]
    for f in traj.temperature
        append!(vals, vec(f[:, layer]))
    end
    return extrema(vals)
end



# Extracts area weights of a grid from field
function area_weights(field::AbstractField)
    grid  = field.grid
    Ω     = get_solid_angles(grid)              # solid angle per ring (length nlat)
    rings = eachring(grid)                       # point-index range per ring
    w     = zeros(Float32, size(field, 1))       # npoints
    for (j, ring) in enumerate(rings)
        @views w[ring] .= Ω[j]
    end
    return w .* (length(w) / sum(w))            # normalize so mean weight = 1
end