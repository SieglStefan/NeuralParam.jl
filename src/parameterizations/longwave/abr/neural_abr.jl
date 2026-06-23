### NeuralABRLW parameterization
###
### - Emulating the AnalyticBandRadiation.jl longwave parameterization



# NeuralABRLW parameterization
struct NeuralABRLW{C,Z,M,P,S,V} <: AbstractABRLW
    n_in::Int           # input dimension of neural network
    n_out::Int          # output -//-

    arch_config::C      # architecture configuration of scheme

    zscore::Z           # loaded zscore parameters

    nn::M               # neural network (Lux)
    ps::P               # parameters of the NN (Lux)
    st::S               # state of the NN (Lux)

    input_buffer::V     # input buffer to avoid allocation
end


# Constructor for creating Lux nn architecture and parameters
function NeuralABRLW(
    spectral_grid::SpectralGrid,
    arch_config;
    zscore_file = nothing,
    rng = Random.default_rng(),
)

    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers
    arch = spectral_grid.architecture

    # Calculate nn input dimensionnlayers
    # - temperature profile:        nlayers
    # - humidity profile:           nlayers
    # - surface_pressure:           1
    # - sea_surface_temperature:    1
    # - land_surface_temperature:   1
    # - land_fraction:              1
    n_in = 2*nlayers + 4

    # Calculate nn output dimension
    # - temperature tendencies:     nlayers
    n_out = nlayers


    # Load zscore statistics
    if isnothing(zscore_file)
        file = "zscore_abrlw_L$(nlayers).jld2"
    else
        file = zscore_file
    end

    zscore = ZScoreStats(file, arch)


    # Create nn architecture
    nn, ps, st = setup_arch(arch_config, n_in, n_out, rng)


    # Create empty input buffer
    input_buffer = zeros(Float32, n_in)


    return NeuralABRLW(
        n_in, n_out,
        arch_config,
        zscore,
        nn, ps, st,
        input_buffer
    )
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::NeuralABRLW, ::PrimitiveEquation)
    return nothing
end


# SpeedyWeather parameterization function for updating temperatureand flux tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    scheme::NeuralABRLW,
    model::SpeedyWeather.AbstractModel,
)

    # Extract number of vertical layers
    nlayers = model.spectral_grid.nlayers


    # Alias input buffer
    X  = scheme.input_buffer

    # Extract input variables into input buffer
    for k in 1:nlayers
        X[k] = vars.grid.temperature[ij,k]
        X[nlayers+k] =  vars.grid.humidity[ij,k]
    end

    X[2*nlayers+1] = vars.grid.pressure[ij]                              
    X[2*nlayers+2] = vars.prognostic.ocean.sea_surface_temperature[ij]                                       
    X[2*nlayers+3] = vars.prognostic.land.soil_temperature[ij,1]
    X[2*nlayers+4] = model.land_sea_mask.mask[ij]


    # Normalize input variables
    X .= zscore.(X, scheme.zscore.input_mean, scheme.zscore.input_std)

    # Lux forward pass
    Y, _ = Lux.apply(scheme.nn, X, scheme.ps, scheme.st)

    # Renormalize output variables
    Y .= inv_zscore.(Y, scheme.zscore.output_mean, scheme.zscore.output_std)


    # Update temperature tendencies
    for k in 1:nlayers
        vars.tendencies.grid.temperature[ij, k] += Y[k]
    end

    return nothing
end

