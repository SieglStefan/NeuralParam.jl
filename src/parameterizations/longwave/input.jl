### Input functions
###
### Utility functions for extracting and preprocessing scheme inputs



### Extracting input functions
# Temperature profile
@inline function in_T(X, o, ij, vars, model, scheme)
    T = SpeedyWeather.get_prognostic_step(vars.grid.temperature, model.time_stepping, scheme)
    for k in axes(T, 2)
        X[o+k] = T[ij,k]
    end
    return o + size(T, 2)
end

# Log10 humidity profile
@inline function in_q(X, o, ij, vars, model, scheme)
    q = SpeedyWeather.get_prognostic_step(vars.grid.humidity, model.time_stepping, scheme)
    for k in axes(q, 2)
        X[o+k] = log10(q[ij,k] + 1f-9)          # add small number to avoid log10(0) = -Inf
    end
    return o + size(q, 2)
end

# Scalar inputs
@inline in_p(X,   o, ij, vars, model, scheme) = (X[o+1] = vars.parameterizations.surface_pressure[ij]; o+1)
@inline in_lat(X, o, ij, vars, model, scheme) = (X[o+1] = model.geometry.latds[ij]; o+1)
@inline in_lf(X,  o, ij, vars, model, scheme) = (X[o+1] = model.land_sea_mask.land_fraction[ij]; o+1)
@inline in_sst(X, o, ij, vars, model, scheme) = (X[o+1] = SpeedyWeather.get_prognostic_step(vars.prognostic.ocean.sea_surface_temperature, model.time_stepping, scheme)[ij]; o+1)
@inline in_lst(X, o, ij, vars, model, scheme) = (X[o+1] = vars.prognostic.land.soil_temperature[ij,1]; o+1)
@inline in_Ts(X,  o, ij, vars, model, scheme) = (X[o+1] = surface_temp(ij, vars, model, scheme); o+1)



# Input dictionary of all available inputs
const INPUTS = (;
    # Profile inputs
        T   = (in_T,   :profile),       # temperature profile
        q   = (in_q,   :profile),       # log10 humidity profile
    # Scalar inputs
        p   = (in_p,   :scalar),        # surface pressure
        lat = (in_lat, :scalar),        # latitude
        lf  = (in_lf,  :scalar),        # land fraction
        sst = (in_sst, :scalar),        # sea surface temperature
        lst = (in_lst, :scalar),        # land surface temperature (top layer)
        Ts  = (in_Ts,  :scalar),        # combined surface temperature
)

# Function for selecting a set of inputs from the dictionary
input_spec(names::Symbol...) = NamedTuple{names}(map(n -> INPUTS[n], names))

# Predefined default selection for different use-cases
const INPUT_NLW_OBLW = input_spec(:T, :Ts, :lat)



# Function for summing the length of the input vector
n_inputs(input_spec, nlayers) = sum((last(e) === :profile ? nlayers : 1) for e in values(input_spec))

# Function for mapping every input to its slice of the input vector, e.g. (; T = 1:8, Ts = 9:9)
function input_layout(input_spec, nlayers)

    offset = 0

    # Returns a vector of ranges by applying "entry -> body" to each entry of input_spec
    return map(input_spec) do entry

        # Length of this input and the range it occupies
        n = last(entry) === :profile ? nlayers : 1
        range = offset+1:offset+n
        offset += n

        return range
    end
end

# Function for filling the input buffer with the selected inputs
#   - @generated unrolls the iteration into straight-line calls at compile time.
#     The previous foldl fetched each input function out of `spec` at runtime, which
#     Enzyme could not resolve statically
@generated function fill_inputs!(X, input_spec::NamedTuple{names}, ij, vars, model, scheme) where {names}

    calls = [:(o = first(input_spec.$n)(X, o, ij, vars, model, scheme)) for n in names]

    return quote
        Base.@_propagate_inbounds_meta
        o = 0
        $(Expr(:block, calls...))
        return o
    end
end


    
# Combined surface temperature, calculated from sea and land surface temperatures and weighted by land fraction
@inline function surface_temp(ij, vars, model, scheme)

    sst = SpeedyWeather.get_prognostic_step(vars.prognostic.ocean.sea_surface_temperature, model.time_stepping, scheme)[ij]
    lst = vars.prognostic.land.soil_temperature[ij,1]
    lf  = model.land_sea_mask.land_fraction[ij]

    Ts_o = ifelse(isfinite(sst), sst, lst)
    Ts_l = ifelse(isfinite(lst), lst, sst)

    return (1 - lf)*Ts_o + lf*Ts_l
end



