### ConstLinearLW parameterization
###
### - Simple baseline scheme using global constant a and b per layer for calculating linear    
###     tendencies dT_k = a_k * T_k + b_k, where k labels vertical layers.
### - a and b vary only in the vertical.
### - Inspired by the Budyko-Sellers model (linearization of Stefan–Boltzmann law)



# ConstLinearLongwave parameterization
struct ConstLinearLW{P} <: AbstractLinearLW
    scaling::Scaling        # scaling factors, used for normalized gradients
    
    ps::P                   # parameters a and b
end


# Convenience constructor
function ConstLinearLW(
    spectral_grid::SpectralGrid;
    user_scaling = nothing
)

    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers

    # Decide scaling (per-layer constant defaults)
    if isnothing(user_scaling)
        scaling = Scaling(nlayers)
    else
        scaling = user_scaling
    end

    # Setup parameter field
    ps = (; a = -ones(Float32, nlayers), b = ones(Float32, nlayers))
    
    return ConstLinearLW(scaling, ps)
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::ConstLinearLW, ::PrimitiveEquation)
    return nothing
end


# SpeedyWeather parameterization function for updating temperature tendencies 
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    scheme::ConstLinearLW,
    model::SpeedyWeather.AbstractModel,
)

    # Extract number of vertical layers
    nlayers = model.spectral_grid.nlayers

    # Update temperature tendencies
    for k in 1:nlayers
        
        ak = scheme.ps.a[k] * scheme.scaling.sc_a[k]
        bk = scheme.ps.b[k] * scheme.scaling.sc_b[k]

        dTk = ak * vars.grid.temperature_prev[ij,k] + bk

        vars.tendencies.grid.temperature[ij,k] += dTk
    end

    return nothing
end