### Constant linear longwave parameterization
###
###
### Simple baseline scheme of the form dT = a*T + b



# ConstLinearLongwave parameterization
@kwdef mutable struct ConstLinearLongwave <: AbstractConstLongwave
    a::Float32 = -6.3f-7        # linear temperature coefficient
    b::Float32 = 1.4f-4         # constant temperature tendency
    sc_a::Float32 = 5f-8        # scaling factor for gradient descent update of a
    sc_b::Float32 = 5f-6        # scaling factor for gradient descent update of b
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::ConstLinearLongwave, ::SpeedyWeather.AbstractModel)
    return nothing
end


# SpeedyWeather parameterization function for updating temperature tendencies 
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    para::ConstLinearLongwave,
    model::SpeedyWeather.AbstractModel,
)

    # Loop over vertical layers and update temperature tendencies
    for k in 1:model.spectral_grid.nlayers
        Tk = vars.grid.temperature[ij, k]
        dTk = para.a * Tk + para.b

        vars.tendencies.grid.temperature[ij, k] += dTk
    end

    return nothing
end