### ConstLinearLW parameterization
###
### - Simple baseline scheme using global constant a and b per for calculating linear    
###     tendencies dT_k = a_k * T_k + b_k, where k indices labels vertical layers.
### - a and b vary only in the vertical.
### - Inspired by the Budyko-Sellers model (linearization of Stefan–Boltzmann law)



# ConstLinearLongwave parameterization
struct ConstLinearLW{P} <: AbstractLinearLW
    ps::P
    config::ConstLinearLWConfig             # parameterization config
end


# Convenience constructor
function ConstLinearLW(config::ConstLinearLWConfig)
    nlayers= length(config.sc_a)
    ps= (; a=zeros(Float32, nlayers), b=zeros(Float32, nlayers))
    
    return ConstLinearLW(ps, config)
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::ConstLinearLW, ::SpeedyWeather.AbstractModel)
    return nothing
end


# SpeedyWeather parameterization function for updating temperature tendencies 
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    para::ConstLinearLW,
    model::SpeedyWeather.AbstractModel,
)

    # Extract number of vertical layers
    nlayers = model.spectral_grid.nlayers

    # Loop over vertical layers and update temperature tendencies
    for k in 1:nlayers
        
        ak = para.ps.a[k] * para.config.sc_a[k]
        bk = para.ps.b[k] * para.config.sc_b[k]

        dTk = ak * vars.grid.temperature[ij,k] + bk

        vars.tendencies.grid.temperature[ij,k] += dTk
    end

    return nothing
end