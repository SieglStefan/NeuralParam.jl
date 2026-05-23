### Structs and functions for the LinearLongwave Parameterization



# Defines a parameterization scheme for linear longwave radiation: dT = a * T + b, where a and b are parameters of the scheme
@kwdef mutable struct LinearLongwave{NF} <: SpeedyWeather.AbstractLongwave
    a::NF = -6.3f-7
    b::NF = 1.4f-4
    sc_grad_a::Float32 = 1f-11            # scaling factor for gradient of a for gradient descent
    sc_grad_b::Float32 = 1f-6             # scaling factor for gradient of b for gradient descent
end


# Convenience constructor
function LinearLongwave(SG::SpeedyWeather.SpectralGrid; kwargs...)
    return LinearLongwave{SG.NF}(; kwargs...)
end


# Initializing function
function SpeedyWeather.initialize!(::LinearLongwave, ::AbstractModel)
    return nothing
end


# Calculate tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(ij, vars::Variables, para::LinearLongwave, model)
    
    # Main loop: Get current temperature of grid-cell ij and update tendencies
    for k in 1:model.spectral_grid.nlayers
        Tk = vars.grid.temperature[ij,k]
        dTk = para.a * Tk + para.b
        vars.tendencies.grid.temperature[ij,k] += dTk
    end

    return nothing
end