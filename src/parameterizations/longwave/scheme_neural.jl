### NeuralLW parameterization
###
### Uses a NN to emulate LW parameterization schemes 



# NeuralLongwave parameterization
struct NeuralLW{N,P,S,C,I,O,Z,B} <: AbstractLW
    nn::N                   # neural network (Lux)
    ps::P                   # parameters of the NN (Lux)
    st::S                   # state of the NN (Lux)

    n_in::Int               # input dim of NN
    n_out::Int              # output dim of NN
    arch_config::C          # architecture configuration of NN  

    inputs::I               # list of inputs used in the scheme, e.g. (:T, :Ts)
    output_form::O          # output form of scheme, e.g. :linear
    zscore::Z               # loaded zscore parameters

    input_buffer::B         # input buffer to avoid allocation

    def_ocean_em::Float32   # default ocean emissivity
    def_land_em::Float32    # default land emissivity
    def_co2::Float32        # default CO2 concentration
end


# Constructor for creating Lux nn architecture and parameters
function NeuralLW(
    spectral_grid::SpectralGrid,
    arch_config,
    inputs,
    output_form,
    zscore_name;
    def_ocean_em = 0.92f0,
    def_land_em = 0.92f0,
    def_co2 = 280f0,
    rng = Random.default_rng(),
)
  
    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers

    # Calculate nn input dimension regarding given inputs
    n_in = n_inputs(inputs, nlayers)

    # Calculate NN output dimension
    # - output_form = :linear:  2*nlayers + 5
    # - output_form = :direct:  nlayers + 2
    n_out = n_outputs(output_form, nlayers)


    # Load zscore statistics 
    zscore = ZScoreStats(zscore_name, inputs, output_form, nlayers)


    # Create nn architecture
    nn, ps, st = setup_arch(arch_config, n_in, n_out, rng)


    # Create empty input buffer
    input_buffer = zeros(Float32, n_in)


    return NeuralLW(
        nn, ps, st,
        n_in, n_out, arch_config,
        inputs, output_form, zscore,
        input_buffer,
        def_ocean_em, def_land_em, def_co2
    )
end


# Helper function for updating parameterization parameters
function update_ps(lw::NeuralLW, ps_new)
    return NeuralLW(
        lw.nn, ps_new, lw.st,
        lw.n_in, lw.n_out, lw.arch_config,
        lw.inputs, lw.output_form, lw.zscore,
        lw.input_buffer,
        lw.def_ocean_em, lw.def_land_em, lw.def_co2
    )
end



# Initializing function for SpeedyWeather (nothing is needed here yet)
function SpeedyWeather.initialize!(::NeuralLW, ::PrimitiveEquation)
    return nothing
end


# SpeedyWeather parameterization function for updating temperature tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(
    ij,
    vars::SpeedyWeather.Variables,
    scheme::NeuralLW,
    model::SpeedyWeather.AbstractModel,
)

    # Populate input buffer
    X = scheme.input_buffer
    fill_inputs!(X, scheme.inputs, ij, vars, model, scheme)


    # Normalize input variables
    X .= zscore.(X, scheme.zscore.input_mean, scheme.zscore.input_std)

    # Lux forward pass
    Y, _ = Lux.apply(scheme.nn, X, scheme.ps, scheme.st)

    # Renormalize output variables
    Y .= inv_zscore.(Y, scheme.zscore.output_mean, scheme.zscore.output_std)


    # Write tendencies
    write_lw!(ij, vars, model, scheme; Y)

    return nothing
end



# Define written info for NeuralLW parameterization scheme
info_scheme(s::NeuralLW) = (;
    scheme       = "NeuralLW",

    n_in         = s.n_in,
    n_out        = s.n_out,
    info_arch(s.arch_config)...,

    inputs       = collect(string.(keys(s.inputs))),
    output_form  = string(nameof(typeof(s.output_form))),
    zscore_stats = s.zscore.zscore_name,

    def_ocean_em = s.def_ocean_em,
    def_land_em  = s.def_land_em,
    def_co2      = s.def_co2,
)