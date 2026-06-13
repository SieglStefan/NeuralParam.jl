### ConstLinearLW parameterization
###
### - Simple baseline scheme using global constant a and b per for calculating linear    
###     tendencies dT_k = a_k * T_k + b_k, where k indices labels vertical layers.
### - a and b vary only in the vertical.
### - Inspired by the Budyko-Sellers model (linearization of Stefan–Boltzmann law)



# ConstLinearLongwave parameterization
struct ConstLinearLW <: AbstractLinearLW
    a::Vector{Float32}                      # linear parameter
    b::Vector{Float32}                      # constant parameter

    config::ConstLinearLWConfig             # parameterization config
end


# Convencience constructor
function ConstLinearLW(
    config::ConstLinearLWConfig
)

    # Choose scaling factors as starting parameters
    return ConstLinearLW(
        config.sc_a,
        config.sc_b,
        config
    )

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

    # Loop over vertical layers and update temperature tendencies
    for k in 1:model.spectral_grid.nlayers
        
        Tk = vars.grid.temperature[ij, k]
        dTk = para.a[k] * Tk + para.b[k]

        vars.tendencies.grid.temperature[ij, k] += dTk
    end

    return nothing
end