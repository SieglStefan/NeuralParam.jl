### General utility functions
###
### Helper functions used across



# Extract vertical layer k from a series of fields
extract_layer(layer, f) = [i[:, layer] for i in f]


# XXX
function target_colorrange(traj; layer)
    vals = Float32[]
    for f in traj.temperature
        append!(vals, vec(f[:, layer]))
    end
    return extrema(vals)
end



steps_from_days(days, Δt_sec) = round(Int, days * 86400 / Δt_sec)