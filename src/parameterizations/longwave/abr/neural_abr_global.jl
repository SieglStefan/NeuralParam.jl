### Global NeuralABRLW parameterization
###
### - Emulating the AnalyticBandRadiation.jl longwave parameterization
### - Built for GPU usage
### - Acts as an global parameterization



# Global NeuralABRLW parameterization
struct NeuralABRLWGlobal{C,Z,M,P,S,F} <: AbstractABRLW
    n_in::Int           # input dimension of neural network
    n_out::Int          # output -//-
    n_points::Int       # number of columns of the grid
    
    arch_config::C      # architecture configuration of nn

    zscore::Z           # loaded zscore parameters

    nn::M               # neural network (Lux)
    ps::P               # parameters of the NN (Lux)
    st::S               # state of the NN (Lux)

    def_co2::F          # default co2 concentration, used if no co2 propagation
    def_ocean_em::F     # default ocean emissivity
    def_land_em::F      # default land emissivity 
end


# Constructor for creating Lux nn architecture and parameters
function NeuralABRLWGlobal(
    spectral_grid::SpectralGrid,
    arch_config;
    zscore_folder = nothing,
    def_co2 = 280f0,
    def_ocean_em = 1f0,
    def_land_em = 1f0,
    rng = Random.default_rng(),
)

    # Get architecutre (cpu vs gpu)
    arch = spectral_grid.architecture
    device = arch isa SpeedyWeather.Architectures.AbstractCPU ? cpu_device() : gpu_device()


    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers

    # Calculate nn input dimension
    # - temperature profile:        nlayers
    # - humidity profile:           nlayers
    # - surface_pressure:           1
    # - co2 concentration:          1
    # - sea_surface_temperature:    1
    # - land_surface_temperature:   1
    # - land_fraction:              1
    # - ocean emissivity:           1
    # - land emissivity:            1
    n_in = 2*nlayers + 7

    # Calculate nn output dimension
    # - temperature tendencies:     nlayers
    # - diag. fluxes:               2
    n_out = nlayers + 2


    # Load zscore statistics
    if isnothing(zscore_folder)
        folder = "zscore_abrlw_L$(nlayers)"
    else
        folder = zscore_folder
    end

    zscore = ZScoreStats(folder, arch)


    # Create nn architecture
    nn, ps, st = setup_arch(arch_config, n_in, n_out, rng)

    ps = device(ps)
    st = device(st)


    return NeuralABRLWGlobal(
        n_in, n_out, spectral_grid.npoints,
        arch_config,
        zscore,
        nn, ps, st,
        def_co2, def_ocean_em, def_land_em,
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
        nothing, nothing, nothing,       # nn, ps, st stripped (not isbits)
        s.def_co2, s.def_ocean_em, s.def_land_em,
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



    ### Extract variables
    T = vars.grid.temperature_prev
    q = vars.grid.humidity_prev
    p = vars.grid.pressure_prev

    if hasproperty(vars.prognostic, :greenhouse_gases) && haskey(vars.prognostic.greenhouse_gases, :co2)
        co2 = vars.prognostic.greenhouse_gases.co2[]
    else
        co2 = scheme.def_co2
    end

    sst = vars.prognostic.ocean.sea_surface_temperature  
    lst = @view vars.prognostic.land.soil_temperature[:,1]
    land_fraction = model.land_sea_mask.mask
    
    ocean_em = scheme.def_ocean_em    # XXX SW does not propagate yet
    land_em = scheme.def_land_em     # XXX SW does not propagate yet



    # Alias input scratch array
    X = vars.parameterizations.nn_input

    # Populate scratch array
    @views begin
        for k in 1:nlayers
            X[k, :]          .= T[:,k]
            X[nlayers+k, :]  .= log10.(q[:,k] .+ 1f-9)
        end

        X[2*nlayers+1, :]  .= p
        X[2*nlayers+2, :]  .= co2
        X[2*nlayers+3, :]  .= sst
        X[2*nlayers+4, :]  .= lst
        X[2*nlayers+5, :]  .= land_fraction
        X[2*nlayers+6, :]  .= ocean_em
        X[2*nlayers+7, :]  .= land_em
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

    # Update diagnostic fluxes from nn
    vars.parameterizations.outgoing_longwave .= @view Y[nlayers+1,:]
    vars.parameterizations.surface_longwave_down .= @view Y[nlayers+2,:]



    # Calculate and update analytical fluxes (same as in ABR, no nn needed)
    σ_SB = model.atmosphere.stefan_boltzmann

    U_sfc_ocean = ifelse.(isfinite.(sst), ocean_em .* σ_SB .* sst.^4, 0f0)
    U_sfc_land  = ifelse.(isfinite.(lst), land_em  .* σ_SB .* lst.^4, 0f0)
    U_sfc_bb    = (1 .- land_fraction) .* U_sfc_ocean .+ land_fraction .* U_sfc_land

    vars.parameterizations.surface_longwave_up .= U_sfc_bb
    vars.parameterizations.ocean.surface_longwave_up .= U_sfc_ocean
    vars.parameterizations.land.surface_longwave_up .= U_sfc_land

    return nothing
end


# Define column parameterization to do nothing
SpeedyWeather.parameterization!(ij, vars, scheme::NeuralABRLWGlobal, model) = nothing
