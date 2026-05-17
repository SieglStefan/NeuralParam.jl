using JLD2

@kwdef struct NeuralLinearLongwaveConfig
    name::String
    width::Int = 32
    n_hidden::Int = 2
    activation::Symbol = :tanh
    sc_a::Float32 = 1f-6
    sc_b::Float32 = 1f-3
    T_mean::Float32 # = XXX
    T_std::Float32 # = XXX
end




# Defines a parameterization scheme for neural linear longwave scheme
@kwdef struct NeuralLinearLongwave{M,P,S,C} <: SpeedyWeather.AbstractLongwave
    nn::M
    ps::P
    st::S
    config::C
end


# Initializing function
function NeuralLinearLongwave(
        SG::SpeedyWeather.SpectralGrid; 
        config = NeuralLinearLongwaveConfig(),
        rng = Random.default_rng())
    
    n_in = SG.nlayers
    n_out = 2*n_in

    width = config.width
    n_hidden = config.n_hidden
    act = config.activation

    hidden = [Lux.Dense(width => width, act) for _ in 1:n_hidden]

    nn = Lux.Chain(
        Lux.Dense(n_in => width, act),
        hidden...,
        Lux.Dense(width => n_out),
    )

    ps, st = Lux.setup(rng, nn)

    return NeuralLinearLongwave(; nn, ps, st, config)
end

function SpeedyWeather.initialize!(::NeuralLinearLongwave, ::AbstractModel)
    return nothing
end

# Calculate the tendencies
Base.@propagate_inbounds function SpeedyWeather.parameterization!(ij, vars::Variables, para::NeuralLinearLongwave, nn)
    
        
    Tij = vars.grid.temperature[ij,:]

    (; nn, ps, st) = para

    Tij_norm = zscore(Tij, para.config.T_mean, para.config.T_std)

    y, _ = Lux.apply(nn, Tij_norm, ps, st)

    a = para.config.sc_a .* y[1:2:end]
    b = para.config.sc_b .* y[2:2:end]

        
    for k in eachindex(Tij)
        vars.tendencies.grid.temperature[ij,k] += a[k] * Tij[k] + b[k]
    end

    return nothing
end




function save(; path::String, para::NeuralLinearLongwave)
    
    mkpath(path)

    filepath = joinpath(path, para.config.name * ".jld2")
    
    jldsave(
        filepath;
        config = para.config,
        ps = para.ps,
        st = para.st,
    )

    return filepath
end


function load_neural_longwave(; path::String, name::String, SG::SpeedyWeather.SpectralGrid)
    
    filepath = joinpath(path, name * ".jld2")
    
    data = JLD2.load(filepath)

    config = data["config"]
    ps = data["ps"]
    st = data["st"]

    tmp = NeuralLinearLongwave(SG; config=config)


    return NeuralLinearLongwave(tmp.nn, ps, st, config)

end