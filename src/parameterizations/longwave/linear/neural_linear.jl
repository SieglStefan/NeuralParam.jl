### NeuralLinearLW parameterization
###
### - Using a neural network for calculating a and b for the linear scheme dT = a*T+b
### - Neural version of ConstLinearLW
### - Inspired by the Budyko-Sellers model (linearization of Stefan–Boltzmann law)



# NeuralLinearLongwave parameterization
struct NeuralLinearLW{C,Z,T,M,P,S,V} <: AbstractLinearLW
    n_in::Int               # input dimension of neural network
    n_out::Int              # output -//-
    
    arch_config::C          # architecture configuration of scheme

    zscore::Z               # loaded zscore parameters
    scaling::T              # loaded scaling parameters

    nn::M                   # neural network (Lux)
    ps::P                   # parameters of the NN (Lux)
    st::S                   # state of the NN (Lux)

    input_buffer::V         # input buffer to avoid allocation
end


# Constructor for creating Lux nn architecture and parameters
function NeuralLinearLW(
    spectral_grid::SpectralGrid,
    arch_config;
    zscore_folder = nothing,    # folder in data/stats containing zscore stats
    scaling_folder = nothing,   # folder in data/stats containing scaling stats
    standard_scaling = false,   # use standard scaling 
    rng = Random.default_rng(),
)
  
    # Extract number of vertical layers and architecture
    nlayers = spectral_grid.nlayers
    arch = spectral_grid.architecture

    # Calculate nn input dimension
    # - temperature profile: nlayers
    n_in = nlayers

    # Calculate NN output dimension
    # - a: nlayers
    # - b: nlayers
    n_out = 2*nlayers


    # Load zscore statistics
    if isnothing(zscore_folder)
        zs_folder = "zscore_llw_L$(nlayers)"
    else
        zs_folder = zscore_folder
    end
    
    zscore = ZScoreStats(zs_folder, arch)


    # Load scaling statistics
    if standard_scaling == false
        if isnothing(scaling_folder)
            sc_folder = "scaling_llw_L$(nlayers)"
        else
            sc_folder = scaling_folder
        end

        scaling = Scaling(sc_folder, arch)
    else
        scaling = Scaling(nlayers)
    end


    # Create nn architecture
    nn, ps, st = setup_arch(arch_config, n_in, n_out, rng)


    # Create empty input buffer
    input_buffer = zeros(Float32, n_in)


    return NeuralLinearLW(
        n_in, n_out,
        arch_config,
        zscore, scaling,
        nn, ps, st,
        input_buffer
    )
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::NeuralLinearLW, ::PrimitiveEquation)
    return nothing
end


# SpeedyWeather parameterization function for updating temperature tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    scheme::NeuralLinearLW,
    model::SpeedyWeather.AbstractModel,
)

    # Extract number of vertical layers
    nlayers = model.spectral_grid.nlayers


    # Extract variables
    T = @view vars.grid.temperature_prev[ij,:] 


    # Populate input buffer
    X  = scheme.input_buffer

    for k in 1:nlayers
        X[k] = T[k]          
    end


    # Normalize input variables
    X .= zscore.(X, scheme.zscore.input_mean, scheme.zscore.input_std)

    # Lux forward pass
    Y, _ = Lux.apply(scheme.nn, X, scheme.ps, scheme.st)


    # Renormalize and update temperature tendencies
    for k in 1:nlayers

        ak = Y[2*k-1] * scheme.scaling.sc_a[k]
        bk = Y[2*k] * scheme.scaling.sc_b[k]

        dTk = ak * T[k] + bk

        vars.tendencies.grid.temperature[ij,k] += dTk
    end

    return nothing
end