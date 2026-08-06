### ConstLW parameterization
###
### Uses global constant parameters to emulate LW parameterization schemes 



# ConstLW parameterization
struct ConstLW{P,O,Z} <: AbstractLW
    ps::P                   # normalized scheme parameters for linearization

    n_out::Int              # number of parameters in the scheme

    output_form::O          # output form of scheme
    zscore::Z               # loaded zscore parameters

    def_ocean_em::Float32   # default ocean emissivity
    def_land_em::Float32    # default land emissivity
    def_co2::Float32        # default CO2 concentration
end


# Constructor for creating Lux nn architecture and parameters
function ConstLW(
    spectral_grid::SpectralGrid,
    output_form,
    zscore_name;
    def_ocean_em = 0.92f0,
    def_land_em = 0.92f0,
    def_co2 = 280f0,
)
  
    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers

    # Calculate scheme parameter number
    # - output_form = :linear:  2*nlayers + 5
    # - output_form = :direct:  nlayers + 2
    n_out = n_outputs(output_form, nlayers)


    # Load zscore statistics and check
    zscore = ZScoreStats(zscore_name, input_spec(;), output_form, nlayers)


    # Setup parameters (zeros: calibrated mean)
    ps = zeros(Float32, n_out)


    return ConstLW(
        ps,
        n_out, 
        output_form, zscore,
        def_ocean_em, def_land_em, def_co2
    )
end


# Helper function for updating parameterization parameters
function update_ps(lw::ConstLW, ps_new)
    return ConstLW(
        ps_new,
        lw.n_out, lw.output_form, lw.zscore,
        lw.def_ocean_em, lw.def_land_em, lw.def_co2
    )
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::ConstLW, ::PrimitiveEquation)
    return nothing
end


# SpeedyWeather parameterization function for updating temperature tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    scheme::ConstLW,
    model::SpeedyWeather.AbstractModel,
)

    # Renormalize parameters
    Y = inv_zscore.(scheme.ps, scheme.zscore.output_mean, scheme.zscore.output_std)

    # Write tendencies
    write_lw!(ij, vars, model, scheme; Y)

    return nothing
end



# Define written info for ConstLW parameterization scheme
info_scheme(s::ConstLW) = (;
    scheme       = "ConstLW",

    ps           = s.ps,

    n_out        = s.n_out,

    output_form  = string(nameof(typeof(s.output_form))),
    zscore_stats = s.zscore.zscore_name,

    def_ocean_em = s.def_ocean_em,
    def_land_em  = s.def_land_em,
    def_co2      = s.def_co2,
)