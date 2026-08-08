### ZeroLW parameterization
###
### Used as "no LW parameterization" parameterization scheme for sanity checks and testing



# ZeroLW parameterization
struct ZeroLW <: AbstractLW
end

# Helper function for updating parameterization parameters
function update_ps(lw::ZeroLW, ps_new)
    return lw
end

# Initializing function for SpeedyWeather
function SpeedyWeather.initialize!(::ZeroLW, ::PrimitiveEquation)
    return nothing
end

# SpeedyWeather parameterization function for updating temperature tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    scheme::ZeroLW,
    model::SpeedyWeather.AbstractModel,
)

    # Write zero to all fluxes
    vars.parameterizations.outgoing_longwave[ij]         = 0f0
    vars.parameterizations.surface_longwave_down[ij]     = 0f0
    vars.parameterizations.ocean.surface_longwave_up[ij] = 0f0
    vars.parameterizations.land.surface_longwave_up[ij]  = 0f0
    vars.parameterizations.surface_longwave_up[ij]       = 0f0

    return nothing
end

# Define written info for ZeroLW parameterization scheme
info_scheme(s::ZeroLW) = (;
    scheme       = "ZeroLW",
)