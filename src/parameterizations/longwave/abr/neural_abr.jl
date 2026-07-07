### NeuralABRLW parameterization
###
### - Emulating the AnalyticBandRadiation.jl longwave parameterization



# NeuralABRLW parameterization
struct NeuralABRLW{C,Z,M,P,S,V,F} <: AbstractABRLW
    n_in::Int           # input dimension of neural network
    n_out::Int          # output -//-

    arch_config::C      # architecture configuration of scheme

    zscore::Z           # loaded zscore parameters

    nn::M               # neural network (Lux)
    ps::P               # parameters of the NN (Lux)
    st::S               # state of the NN (Lux)

    input_buffer::V     # input buffer to avoid allocation

    def_co2::F          # default co2 concentration, used if no co2 propagation
    def_ocean_em::F     # default ocean emissivity
    def_land_em::F      # default land emissivity 
end


# Constructor for creating Lux nn architecture and parameters
function NeuralABRLW(
    spectral_grid::SpectralGrid,
    arch_config;
    zscore_folder = nothing,
    def_co2 = 280f0,
    def_ocean_em = 1f0,
    def_land_em = 1f0,
    rng = Random.default_rng(),
)

    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers
    arch = spectral_grid.architecture

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


    # Create empty input buffer
    input_buffer = zeros(Float32, n_in)


    return NeuralABRLW(
        n_in, n_out,
        arch_config,
        zscore,
        nn, ps, st,
        input_buffer,
        def_co2, def_ocean_em, def_land_em,
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



    # Extract variables
    T = @view vars.grid.temperature_prev[ij,:]
    q = @view vars.grid.humidity_prev[ij,:]
    p = vars.grid.pressure_prev[ij]

    if hasproperty(vars.prognostic, :greenhouse_gases) && haskey(vars.prognostic.greenhouse_gases, :co2)
        co2 = vars.prognostic.greenhouse_gases.co2[]
    else
        co2 = scheme.def_co2
    end

    sst = vars.prognostic.ocean.sea_surface_temperature[ij]   
    lst = vars.prognostic.land.soil_temperature[ij,1]
    land_fraction = model.land_sea_mask.mask[ij]
    
    ocean_em = scheme.def_ocean_em    # XXX SW does not propagate yet
    land_em = scheme.def_land_em     # XXX SW does not propagate yet



    # Alias input buffer
    X  = scheme.input_buffer

    # Populate scratch array
    for k in 1:nlayers
        X[k] = T[k]
        X[nlayers+k] = log10(q[k] + 1f-9)
    end

    X[2*nlayers+1] = p                            
    X[2*nlayers+2] = co2                                       
    X[2*nlayers+3] = sst
    X[2*nlayers+4] = lst
    X[2*nlayers+5] = land_fraction
    X[2*nlayers+6] = ocean_em  
    X[2*nlayers+7] = land_em



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

    # Update diagnostic fluxes from nn
    vars.parameterizations.outgoing_longwave[ij] = Y[nlayers+1]
    vars.parameterizations.surface_longwave_down[ij] = Y[nlayers+2]



    # Calculate and update analytical fluxes (same as in ABR, no nn needed)
    σ_SB = model.atmosphere.stefan_boltzmann

    U_sfc_ocean = ifelse(isfinite(sst), ocean_em * σ_SB * sst^4, 0f0)
    U_sfc_land  = ifelse(isfinite(lst), land_em  * σ_SB * lst^4, 0f0)
    U_sfc_bb    = (1 - land_fraction) * U_sfc_ocean + land_fraction * U_sfc_land

    vars.parameterizations.surface_longwave_up[ij] = U_sfc_bb
    vars.parameterizations.ocean.surface_longwave_up[ij] = U_sfc_ocean
    vars.parameterizations.land.surface_longwave_up[ij] = U_sfc_land

    return nothing
end

