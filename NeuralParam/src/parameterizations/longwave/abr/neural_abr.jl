### NeuralABRLW parameterization
###
### - Emulating the AnalyticBandRadiation.jl longwave parameterization



# NeuralABRLW parameterization object
struct NeuralABRLW{M,P,S,C,V} <: AbstractABRLW
    nn::M               # neural network (Lux)
    ps::P               # parameters of the NN (Lux)
    st::S               # state of the NN (Lux)
    config::C           # configuration of the NN
    input_buffer::V     # input buffer to avoid allocation
end


# Constructor for creating Lux NN architecture and parameters
function NeuralABRLW(
    config::NeuralABRLWConfig;
    rng = Random.default_rng(),
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

    return NeuralABRLW(nn, ps, st, config, input_buffer)
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::NeuralABRLW, ::SpeedyWeather.AbstractModel)
    return nothing
end


# SpeedyWeather parameterization function for updating temperature tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    para::NeuralABRLW,
    model::SpeedyWeather.AbstractModel,
)

    # Extract number of vertical layers
    nlayers = model.spectral_grid.nlayers


    # XXX Extract and normalize NN input variables  
    for k in 1:nlayers
        
        para.input_buffer[k] =          zscore(vars.grid.temperature[ij,k],                         # temperature
                                            para.config.zscore.input_mean[k], 
                                            para.config.zscore.input_std[k])

        para.input_buffer[nlayers+k] =  zscore(vars.grid.humidity[ij,k],                            # humidity
                                            para.config.zscore.input_mean[nlayers+k], 
                                            para.config.zscore.input_std[nlayers+k])
    end

    para.input_buffer[2*nlayers+1] =    zscore(vars.grid.pressure[ij],                              # surface pressure
                                            para.config.zscore.input_mean[2*nlayers+1], 
                                            para.config.zscore.input_std[2*nlayers+1])
                                            
    para.input_buffer[2*nlayers+2] =    zscore(vars.prognostic.ocean.sea_surface_temperature[ij],   # sea surface temperature
                                            para.config.zscore.input_mean[2*nlayers+2], 
                                            para.config.zscore.input_std[2*nlayers+2])
                                            
    para.input_buffer[2*nlayers+3] =    zscore( vars.prognostic.land.soil_temperature[ij,1],        # land surface temperature
                                            para.config.zscore.input_mean[2*nlayers+3], 
                                            para.config.zscore.input_std[2*nlayers+3])                                            

    para.input_buffer[2*nlayers+4] =    zscore(model.land_sea_mask.mask[ij],                        # land fraction
                                            para.config.zscore.input_mean[2*nlayers+4], 
                                            para.config.zscore.input_std[2*nlayers+4])
    
    
    # Lux forward pass
    y, _ = Lux.apply(
        para.nn,
        para.input_buffer,
        para.ps,
        para.st
    )

    # XXX Renormalize and update temperature tendencies
    for k in 1:nlayers

        mean_k = para.config.zscore.output_mean[k]
        std_k = para.config.zscore.output_std[k]

        vars.tendencies.grid.temperature[ij, k] += inv_zscore(y[k], mean_k, std_k)
    end

    return nothing
end

