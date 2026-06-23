### Global NeuralABRLW parameterization
###
### - Emulating the AnalyticBandRadiation.jl longwave parameterization
### - Supports GPU usage
### - Acts as an global parameterization



# Global NeuralABRLW parameterization
struct NeuralABRLWGlobal{C,Z,M,P,S} <: AbstractABRLW
    n_in::Int           # input dimension of neural network
    n_out::Int          # output -//-
    n_points::Int       # number of columns of the grid
    
    arch_config::C      # architecture configuration of nn

    zscore::Z           # loaded zscore parameters

    nn::M               # neural network (Lux)
    ps::P               # parameters of the NN (Lux)
    st::S               # state of the NN (Lux)
end


# Constructor for creating Lux nn architecture and parameters
function NeuralABRLWGlobal(
    spectral_grid::SpectralGrid,
    arch_config;
    zscore_file = nothing,
    rng = Random.default_rng(),
)

    # Get architecutre (cpu vs gpu)
    arch = spectral_grid.architecture
    device = arch isa SpeedyWeather.Architectures.AbstractCPU ? cpu_device() : gpu_device()


    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers

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

    ps = device(ps)
    st = device(st)


    return NeuralABRLWGlobal(
        n_in, n_out, spectral_grid.npoints,
        arch_config,
        zscore,
        nn, ps, st
    )
end



# Define nn input scratch array to avoid allocation
function SpeedyWeather.variables(scheme::NeuralABRLWGlobal)
    
    return (
        # First define scratch usually created by longwave radiation parameterization
        ParameterizationVariable(:surface_longwave_down, SpeedyWeather.Grid2D(),
            desc = "Surface longwave radiation down", units = "W/m^2"),
        ParameterizationVariable(:surface_longwave_up, SpeedyWeather.Grid2D(),
            desc = "Surface longwave up over ocean", units = "W/m^2", namespace = :ocean),
        ParameterizationVariable(:surface_longwave_up, SpeedyWeather.Grid2D(),
            desc = "Surface longwave up over land", units = "W/m^2", namespace = :land),
        ParameterizationVariable(:surface_longwave_up, SpeedyWeather.Grid2D(),
            desc = "Surface longwave up", units = "W/m^2"),
        ParameterizationVariable(:outgoing_longwave, SpeedyWeather.Grid2D(),
            desc = "Top-of-atmosphere longwave up", units = "W/m^2"),

        # Now create nn input scratch array, which we actually use
        ParameterizationVariable(:nn_input, SpeedyWeather.MatrixDim(scheme.n_in, scheme.n_points),
            desc = "NN input buffer (features × grid points)"),
    )
end



# Defines how a NeuralABRLWGlobal parameterization is converted to "to" 
# - used by SW in column parameterization, which kernel arguments must be "isbits"
# - "isbits" are "pure" data, that means no pointer etc.
# - a Lux nn is not isbit, therefore cannot be passed
function Adapt.adapt_structure(to, s::NeuralABRLWGlobal)
    
    # Return adapted NeuralABRLWGlobal
    return NeuralABRLWGlobal(
        s.n_in, s.n_out, s.n_points,    # isbits
        s.arch_config,                  # isbits
        nothing,                        # zscore (column kernel no-op doesn't use it)
        nothing, nothing, nothing       # nn, ps, st stripped (not isbits)
    )
end


# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::NeuralABRLWGlobal, ::PrimitiveEquation) 
    return nothing
end


# SpeedyWeather parameterization function for updating temperatureand flux tendencies
function SpeedyWeather.parameterization!(
    vars::SpeedyWeather.Variables, 
    scheme::NeuralABRLWGlobal, 
    model::PrimitiveEquation
)

    # Extract number of vertical layers
    nlayers = model.spectral_grid.nlayers


    # Alias input scratch array
    X = vars.parameterizations.nn_input

    # Extract input variables into scratch array
    @views begin
        for k in 1:nlayers
            X[k, :]          .= vars.grid.temperature[:,k]
            X[nlayers+k, :]  .= vars.grid.humidity[:,k]
        end

        X[2*nlayers+1, :]  .= vars.grid.pressure
        X[2*nlayers+2, :]  .= vars.prognostic.ocean.sea_surface_temperature
        X[2*nlayers+3, :]  .= vars.prognostic.land.soil_temperature[:,1]
        X[2*nlayers+4, :]  .= model.land_sea_mask.mask
    end


    # Normalize input variables (scalar zscore broadcast over the buffer)
    X .= zscore.(X, scheme.zscore.input_mean, scheme.zscore.input_std)

    # Lux forward pass
    Y, _ = Lux.apply(scheme.nn, X, scheme.ps, scheme.st)

    # Renormalize output variables
    Y .= inv_zscore.(Y, scheme.zscore.output_mean, scheme.zscore.output_std)


    # Extract array out of field
    dT = parent(vars.tendencies.grid.temperature)

    # Update temperature tendencies
    @views for k in 1:nlayers
        dT[:,k] .+= Y[k,:]
    end

    return nothing
end


# Define column parameterization to do nothing
SpeedyWeather.parameterization!(ij, vars, scheme::NeuralABRLWGlobal, model) = nothing
