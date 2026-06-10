### NeuralLinearLongwave parameterization 
###
### Lux-based linear longwave parameterization using (a_k, b_k) = f_NN(T_k) 
### for upgrading tendencies dTk = a_k * T_k + b_k.
###
### Important:
### - TODO Uses Reactant to speed up usage of the NN



# NeuralLinearLongwave parameterization object
@kwdef mutable struct NeuralLinearLongwave{M,P,S,C} <: AbstractLuxLongwave
    nn::M               # neural network (Lux)
    ps::P               # parameters of the NN (Lux)
    st::S               # state of the NN (Lux)
    config::C           # configuration of the NN
end


# Constructor for creating Lux NN architecture and parameters
function NeuralLinearLongwave(
    config::NeuralLinearLongwaveConfig;
    rng = Random.default_rng(),
)

    # Build Lux model and Lux parameter structure    
    nn, ps, st = setup_nn(config, rng)

    return NeuralLinearLongwave(nn, ps, st, config)
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::NeuralLinearLongwave, ::SpeedyWeather.AbstractModel)
    return nothing
end


# Calculate tendencies using Lux
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    para::NeuralLinearLongwave,
    model::SpeedyWeather.AbstractModel,
)

    # Extract NN parameters
    (; nn, ps, st) = para
    nlayers = model.spectral_grid.nlayers


    # Extract NN input variables
    x_raw = Float32[
        vars.grid.temperature[ij, k] for k in 1:nlayers       # temperature
    ]   
    
    @assert length(x_raw) == length(para.config.input_mean) "input length mismatch"

    # Normalize NN input
    x = normalize_nn_input(para, x_raw)


    # Lux forward pass
    y, _ = Lux.apply(nn, x, ps, st)

    # Transform NN output back to physical units
    a, b = unscale_nn_output(para, y)


    # Update temperature tendencies
    for k in 1:nlayers
        vars.tendencies.grid.temperature[ij,k] += a[k] * vars.grid.temperature[ij,k] + b[k]
    end

    return nothing
end