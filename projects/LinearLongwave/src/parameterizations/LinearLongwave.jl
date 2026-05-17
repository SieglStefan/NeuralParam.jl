

# Defines a parameterization scheme for linear longwave radiation: dT = a * T + b, where a and b are parameters of the scheme
@kwdef struct LinearLongwave{NF} <: SpeedyWeather.AbstractLongwave
    a::NF = -1e-6
    b::NF = 1e-3
end

# Convenience constructor
function LinearLongwave(SG::SpeedyWeather.SpectralGrid; kwargs...)
    return LinearLongwave{SG.NF}(; kwargs...)
end

# Initializing function
function SpeedyWeather.initialize!(::LinearLongwave, ::AbstractModel)
    return nothing
end

# Calculate the tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(ij, vars::Variables, para::LinearLongwave, model)
    
    for k in 1:model.spectral_grid.nlayers
        Tk = vars.grid.temperature[ij,k]
        dTk = para.a * Tk + para.b
        vars.tendencies.grid.temperature[ij,k] += dTk
    end

    return nothing
end



