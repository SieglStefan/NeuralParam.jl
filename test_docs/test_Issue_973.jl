# Code from https://github.com/SpeedyWeather/SpeedyWeather.jl/issues/973, adapted





### 1. Create a Lux NN
using Lux, Reactant, Random
rng = Random.default_rng()
lux_model = Chain(Dense(4, 16, leakyrelu), Dense(16, 16, leakyrelu), Dense(16, 1))
parameters, states = Lux.setup(rng, lux_model)



#XXX Here one should convert the array to an ReactantRArray
#x = rand(Float32, 4)
x = Reactant.to_rarray(rand(Float32, 4))



lux_model_forward = @compile lux_model(x, parameters, Lux.testmode(states))





### 2. Define a parameterization with a Lux NN
using SpeedyWeather

struct MLAlbedo{P, S, F} <: SpeedyWeather.AbstractAlbedo
    parameters::P
    states::S
    forward::F
end

SpeedyWeather.initialize!(::MLAlbedo, ::PrimitiveEquation) = nothing                                    # changed to PrimitiveEquation ("i" was missing)

Base.@propagate_inbounds function SpeedyWeather.albedo!(ij, albedo, vars, scheme::MLAlbedo, model)      # changed to new variable system



    #XXX Here again, one need to use an ReactantRArray
    #x = Vector{Float32}(undef, 4)       
    x = Reactant.to_rarray(rand(Float32, 4))
    


    y, _ = scheme.forward(x, scheme.parameters, Lux.testmode(scheme.states))

    albedo[ij] = 0.3                                                                                    # changed to new variable system                                      
    return nothing
end


ml_albedo = MLAlbedo(parameters, states, lux_model_forward)





### 3. Create a SpeedyWeather simulation with NN albedo
spectral_grid = SpectralGrid()
albedo = OceanLandAlbedo(land=ml_albedo, ocean=OceanSeaIceAlbedo(spectral_grid))
model = PrimitiveWetModel(spectral_grid, albedo=albedo)
simulation = initialize!(model)





### 4. Benchmark
using BenchmarkTools
vars, model = SpeedyWeather.unpack(simulation)                                                          # changed to new variable system



#XXX Here one need to change, dependend what to test
#print("With NN: ")                      # 4. Benchmark              (use of normal array)
#print("WithOUT NN: ")                   # 5. Compare to baseline    (comment out x and y in line 43/44 and 48)
#print("With NN (and Reactant): ")       # Comparison                (use of ReactantRArray for x)
@btime SpeedyWeather.parameterization_tendencies!($vars, $model)                                        # changed to new variable system





# Function for testing the output of lux_model_forward:
    # if compiled with normal array: lux_model_forward leads always to the same output
    # if compiled with ReactantRArray: lux_model_forward leads to different output
function test()
    x = Vector{Float32}(undef, 4)                   # take this if compiled with normal array
    #x = Reactant.to_rarray(rand(Float32, 4))       # take this if compiled with ReactantRArray
    y, _ = lux_model_forward(x, parameters, Lux.testmode(states))
    @show x
    @show y
    return nothing
end