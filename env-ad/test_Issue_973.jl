# Code from https://github.com/SpeedyWeather/SpeedyWeather.jl/issues/973, adapted

using Lux, Reactant, Random

# I guess one could just a zero or identity initializer here, see Lux.WeightInitializers
rng = Random.default_rng()

# a small dense NN
lux_model = Chain(Dense(4, 16, leakyrelu), Dense(16, 16, leakyrelu), Dense(16, 1))

# a Lux "model" doesn't actually contain the data, that data is allocated here
# we skip the |> device here as we just evaluate this on the CPU
parameters, states = Lux.setup(rng, lux_model)

# let Reactant compile this model with some dummy input data x
# vector of 4 to match size of NN's first layer



#XXX
x = rand(Float32, 4)


x = Reactant.to_rarray(rand(Float32, 4))
lux_model_forward = @compile lux_model(x, parameters, Lux.testmode(states))



using SpeedyWeather

# define a new type with the objects from the NN we want to pass around
# don't think the forward::F is actually needed given it's a function it could
# probably also exist globally? But not sure how Reactant functions are
# different from Julia functions
struct MLAlbedo{P, S, F} <: SpeedyWeather.AbstractAlbedo
    parameters::P
    states::S
    forward::F
end

# add a generator function here that adapts the device of the NN to the architecture of the spectral grid

# nothing to initialize here
SpeedyWeather.initialize!(::MLAlbedo, ::PrimitiveEquation) = nothing        ### changed to PrimitiveEquation

# extend SpeedyWeather's albedo call with our new MLAlbedo type
# that'll be called for every grid call ij
Base.@propagate_inbounds function SpeedyWeather.albedo!(ij, albedo, vars, scheme::MLAlbedo, model)      ### changed function signature



    #XXX
    x = Vector{Float32}(undef, 4)
    #x = Reactant.to_rarray(rand(Float32, 4))
    


    # this will also allocate the (intermediate?) + return array y, maybe Lux allows to this all in-place
    y, _ = scheme.forward(x, scheme.parameters, Lux.testmode(scheme.states))

    # don't actually do anything with y, just set albedo to 0.3 here
    albedo[ij] = 0.3        ### changed to field in function signature
    return nothing
end

ml_albedo = MLAlbedo(parameters, states, lux_model_forward)

spectral_grid = SpectralGrid()

# use NN albedo over land
albedo = OceanLandAlbedo(land=ml_albedo, ocean=OceanSeaIceAlbedo(spectral_grid))

# pass on to model constructor
model = PrimitiveWetModel(spectral_grid, albedo=albedo)
simulation = initialize!(model)

using BenchmarkTools
vars, model = SpeedyWeather.unpack(simulation)      ### changed



#XXX
#print("With NN and RArray: ")
#@btime $SpeedyWeather.parameterization_tendencies!($vars, $model)




function test()
    x = Vector{Float32}(undef, 4)
    #x = Reactant.to_rarray(rand(Float32, 4))
    y, _ = lux_model_forward(x, parameters, Lux.testmode(states))
    @show x
    @show y
    return nothing
end

test()