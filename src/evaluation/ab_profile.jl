### Functions for a/b profile evaluation of ConstLinearLW schemes
###
### XXX

# Effective heating (dT) of a ConstLinearLW scheme for a given temperature
effective_heating(s::ConstLinearLW, T) = @. s.ps.a * s.scaling.sc_a * T + s.ps.b * s.scaling.sc_b


# Plot a and b vertical profiles side by side
function plot_ab_profiles(schemes::NamedTuple; nlayers, title = "")
    layers = 1:nlayers

    p_a = Plots.plot(; xlabel = "a", ylabel = "Layer", yflip = true, yticks = layers, legend = :topleft)
    p_b = Plots.plot(; xlabel = "b", ylabel = "Layer", yflip = true, yticks = layers, legend = :topleft)

    for (name, s) in pairs(schemes)
        s isa ConstLinearLW || continue                 # skip nothing / non-ab schemes
        Plots.plot!(p_a, s.ps.a, layers; label = String(name), marker = :circle, lw = 2)
        Plots.plot!(p_b, s.ps.b, layers; label = String(name), marker = :circle, lw = 2)
    end

    return Plots.plot(p_a, p_b; layout = (1, 2), plot_title = title)
end


# Plot effective heating dT for a typical temperature profile
function plot_dT_profile(schemes::NamedTuple; nlayers, title = "",
                         T_typical = collect(range(210f0, 290f0, length = nlayers)))
    layers = 1:nlayers
    sec_per_day = 86_400f0

    p = Plots.plot(; xlabel = "dT [K/day]", ylabel = "Layer", yflip = true, yticks = layers, legend = :topleft)
    for (name, s) in pairs(schemes)
        s isa ConstLinearLW || continue
        dT = effective_heating(s, T_typical) .* sec_per_day
        Plots.plot!(p, dT, layers; label = String(name), marker = :circle, lw = 2)
    end
    return p
end