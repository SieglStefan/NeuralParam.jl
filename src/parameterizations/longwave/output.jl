### Output functions
###
### Utility functions for handling scheme outputs. 
### - a new output form needs six methods, all defined here:
###     n_outputs, output_group, output_keys, decode!, predictor, output_stats



### Output forms

# Define output form struct for parameterizations
struct DirectOutput end
struct LinearOutput end
struct PlanckOutput end

# Define output vector length for different output types
n_outputs(::DirectOutput, nlayers) = nlayers + 2                # dT, olw, slw
n_outputs(::LinearOutput, nlayers) = 2*nlayers + 5              # 2*nlayers for dT, 5 for fluxes (olw, slwd)
n_outputs(::PlanckOutput, nlayers) = 2*nlayers + 5              # 2*nlayers for dT, 5 for fluxes (olw, slwd)

# Utility functions for ouput forms
output_group(::DirectOutput) = :direct
output_group(::LinearOutput) = :linear
output_group(::PlanckOutput) = :planck

# Utility functions for output keys
output_keys(::DirectOutput) = (:dT, :olw, :slwd)                # dT(1:n), olw, slwd
output_keys(::LinearOutput) = (:a, :b, :c, :d, :e, :f, :g)      # dT: a(1:n), b(n+1:2n), olw: c, d, e, slwd: f, g
output_keys(::PlanckOutput) = (:a, :b, :c, :d, :e, :f, :g)      # dT: a(1:n), b(n+1:2n), olw: c, d, e, slwd: f, g



### Decoding

# Layer whose temperature the olw coefficients use (also used in stats generation)
mid_layer(nlayers) = nlayers ÷ 2


# Define decodings of output vector Y
@inline function decode!(::DirectOutput, dTdt, Y, state)

    (; nlayers) = state

    # Accumulate temperature tendencies
    for k in 1:nlayers
        dTdt[k] += Y[k]
    end

    return Y[nlayers+1], Y[nlayers+2]
end

@inline function decode!(::LinearOutput, dTdt, Y, state)

    (; T_prof, Ts, nlayers) = state

    # Accumulate temperature tendencies
    for k in 1:nlayers
        dTdt[k] += Y[k]*T_prof[k] + Y[nlayers+k]
    end

    # Accumulate fluxes
    olw  = Y[2*nlayers+1] + Y[2*nlayers+2]*T_prof[mid_layer(nlayers)] + Y[2*nlayers+3]*Ts
    slwd = Y[2*nlayers+4] + Y[2*nlayers+5]*T_prof[nlayers]

    return olw, slwd
end

@inline function decode!(::PlanckOutput, dTdt, Y, state)

    (; T_prof, Ts, nlayers) = state

    # Accumulate temperature tendencies
    for k in 1:nlayers
        dTdt[k] += Y[k]*T_prof[k]^4 + Y[nlayers+k]
    end

    # Accumulate fluxes
    olw  = Y[2*nlayers+1] + Y[2*nlayers+2]*T_prof[mid_layer(nlayers)]^4 + Y[2*nlayers+3]*Ts^4
    slwd = Y[2*nlayers+4] + Y[2*nlayers+5]*T_prof[nlayers]^4

    return olw, slwd
end



### Writing

# Assemble the state a decode! reads from
@inline function lw_state(ij, vars, model, scheme)

    nlayers = model.spectral_grid.nlayers
    T_prof  = @view SpeedyWeather.get_prognostic_step(vars.grid.temperature, model.time_stepping, scheme)[ij,:]

    return (;
        T_prof  = T_prof,                                       # temperature profile
        Ts      = surface_temp(ij, vars, model, scheme),        # combined surface temperature
        lat     = model.geometry.latds[ij],                     # latitude
        lf      = model.land_sea_mask.land_fraction[ij],        # land fraction
        nlayers = nlayers,                                      # number of vertical layers
        σ_SB    = model.atmosphere.stefan_boltzmann,            # Stefan-Boltzmann constant
    )
end


# Function for writing longwave tendencies and fluxes
@inline function write_lw!(ij, vars, model, scheme; Y)

    ### Assemble the state, decode the output vector and write temperature tendencies
    state = lw_state(ij, vars, model, scheme)

    dTdt = @view SpeedyWeather.get_tendency_step(vars.tendencies.grid.temperature, model.time_stepping, scheme)[ij,:]

    olw, slwd = decode!(scheme.output_form, dTdt, Y, state)


    ### Write NN flux tendencies
    vars.parameterizations.outgoing_longwave[ij] = olw
    vars.parameterizations.surface_longwave_down[ij] = slwd


    ### Write analytic flux tendencies
    sst = SpeedyWeather.get_prognostic_step(vars.prognostic.ocean.sea_surface_temperature, model.time_stepping, scheme)[ij]
    lst = vars.prognostic.land.soil_temperature[ij,1]
    ocean_em = scheme.def_ocean_em
    land_em  = scheme.def_land_em

    U_sfc_ocean = ifelse(isfinite(sst), ocean_em * state.σ_SB * sst^4, 0f0)
    U_sfc_land  = ifelse(isfinite(lst), land_em  * state.σ_SB * lst^4, 0f0)
    U_sfc_bb    = (1 - state.lf) * U_sfc_ocean + state.lf * U_sfc_land

    vars.parameterizations.ocean.surface_longwave_up[ij] = U_sfc_ocean
    vars.parameterizations.land.surface_longwave_up[ij] = U_sfc_land
    vars.parameterizations.surface_longwave_up[ij] = U_sfc_bb

    return nothing
end



### Statistics

# Predictor transform of an affine output form: what the fitted coefficients multiply
predictor(::LinearOutput, x) = x
predictor(::PlanckOutput, x) = x.^4


# Shared fit for affine output forms (dT = a*P(T) + b, olw = c + d*P(T[mid_layer]) + e*P(Ts),
#                                     slwd = f + g*P(T[nlayers])), fitted per column
function affine_stats(output_form, samples)

    npoints, nlayers, _ = size(samples.T)

    # Transform the predictors of the output form (LinearOutput: T, PlanckOutput: T^4)
    T  = predictor(output_form, samples.T)
    Ts = predictor(output_form, samples.Ts)

    # Temperature tendency coefficients, one fit per column and layer
    a = zeros(Float32, npoints, nlayers)
    b = zeros(Float32, npoints, nlayers)

    for ij in 1:npoints, k in 1:nlayers
        b[ij,k], a[ij,k] = fit_linear(view(T, ij, k, :), view(samples.dT, ij, k, :))
    end

    # Flux coefficients, one fit per column
    c = zeros(Float32, npoints); d = zeros(Float32, npoints); e = zeros(Float32, npoints)
    f = zeros(Float32, npoints); g = zeros(Float32, npoints)

    for ij in 1:npoints
        c[ij], d[ij], e[ij] = fit_linear(view(T, ij, mid_layer(nlayers), :), view(Ts, ij, :), view(samples.olw, ij, :))
        f[ij], g[ij]        = fit_linear(view(T, ij, nlayers, :), view(samples.slwd, ij, :))
    end

    return (;
        a = mean_std_layers(a), b = mean_std_layers(b),
        c = mean_std(c), d = mean_std(d), e = mean_std(e),
        f = mean_std(f), g = mean_std(g),
    )
end


# Define zscore statistics of the output coefficients, one method per output form
output_stats(::DirectOutput, samples) = (;
    dT   = mean_std_layers(samples.dT),
    olw  = mean_std(samples.olw),
    slwd = mean_std(samples.slwd),
)

output_stats(output_form::LinearOutput, samples) = affine_stats(output_form, samples)
output_stats(output_form::PlanckOutput, samples) = affine_stats(output_form, samples)
