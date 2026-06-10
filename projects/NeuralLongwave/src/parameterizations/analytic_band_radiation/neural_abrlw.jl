### NeuralABRLongwave  parameterization 
###
### Lux-based longwave parameterization emulating AnalyticBandRadiation.jl
###
### Important:
### - TODO Uses Reactant to speed up usage of the NN



# NeuralABRLongwave parameterization object
@kwdef mutable struct NeuralABRLongwave{M,P,S,C} <: AbstractLuxLongwave
    nn::M               # neural network (Lux)
    ps::P               # parameters of the NN (Lux)
    st::S               # state of the NN (Lux)
    config::C           # configuration of the NN
end


# Constructor for creating Lux NN architecture and parameters
function NeuralABRLongwave(
    config::NeuralABRLongwaveConfig;
    rng = Random.default_rng(),
)

    # Build Lux model and Lux parameter structure    
    nn, ps, st = setup_nn(config, rng)

    return NeuralABRLongwave(nn, ps, st, config)
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::NeuralABRLongwave, ::SpeedyWeather.AbstractModel)
    return nothing
end


# Calculate tendencies using Lux
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    para::NeuralABRLongwave,
    model::SpeedyWeather.AbstractModel,
)

    # Extract NN parameters
    (; nn, ps, st) = para
    nlayers = model.spectral_grid.nlayers


    # Extract NN input variables
    x_raw = Float32(vcat(
        [vars.grid.temperature[ij,k] for k=1:nlayers],          # temperature
        [vars.grid.humidity[ij,k] for k=1:nlayers],             # humidity
        vars.grid.pressure[ij],                                 # surface pressure
        vars.prognostic.ocean.sea_surface_temperature[ij],      # sea surface temperature
        vars.prognostic.land.soil_temperature[ij,1],            # land surface temperature
        model.land_sea_mask.mask[ij]                            # land fraction
    ))

    @assert length(x_raw) == length(para.config.input_mean) "input length mismatch"

    # Normalize NN input
    x = normalize_nn_input(para, x_raw)
    

    # Lux forward pass
    y, _ = Lux.apply(nn, x, ps, st)

    # Transform NN output back to physical units
    y_norm = unscale_nn_output(para, y)


    # Update temperature tendencies
    for k in 1:nlayers
        vars.tendencies.grid.temperature[ij, k] += y_norm[k]
    end

    return nothing
end

