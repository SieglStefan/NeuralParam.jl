### NeuralLinearLW parameterization
###
### - Using a neural network for calculating a and b for the linear scheme dT = a*T+b
### - Neural version of ConstLinearLW
### - Inspired by the Budyko-Sellers model (linearization of Stefan–Boltzmann law)



# NeuralLinearLongwave parameterization
struct NeuralLinearLW{M,P,S,C,V} <: AbstractLinearLW
    nn::M               # neural network (Lux)
    ps::P               # parameters of the NN (Lux)
    st::S               # state of the NN (Lux)
    config::C           # configuration of the NN
    input_buffer::V     # input buffer to avoid allocation
end


# Constructor for creating Lux NN architecture and parameters
function NeuralLinearLW(
    config::NeuralLinearLWConfig;
    rng = Random.default_rng()
)
  
    # Create nn architecture
    nn, ps, st = setup_nn(
        config.nn_config,
        config.n_in, 
        config.n_out,
        rng
    )

    # Create input buffer
    input_buffer = zeros(Float32, config.n_in)

    return NeuralLinearLW(nn, ps, st, config, input_buffer)
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::NeuralLinearLW, ::SpeedyWeather.AbstractModel)
    return nothing
end


# SpeedyWeather parameterization function for updating temperature tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    para::NeuralLinearLW,
    model::SpeedyWeather.AbstractModel,
)

    # Extract number of vertical layers
    nlayers = model.spectral_grid.nlayers


    # Extract and normalize NN input variables
    for k in 1:nlayers

        para.input_buffer[k] = zscore(vars.grid.temperature[ij,k],          # temperature
                                        para.config.zscore.input_mean[k], 
                                        para.config.zscore.input_std[k])                        
    end


    # Lux forward pass
    y, _ = Lux.apply(
        para.nn,
        para.input_buffer,
        para.ps,
        para.st
    )


    # Renormalize and update temperature tendencies
    for k in 1:nlayers

        ak = y[2*k-1] * para.config.sc_a[k]
        bk = y[2*k] * para.config.sc_b[k]

        vars.tendencies.grid.temperature[ij,k] += ak * vars.grid.temperature[ij,k] + bk
    end

    return nothing
end