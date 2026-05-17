using SpeedyWeather, CairoMakie, Random

include("LinearLongwave.jl")
include("loss_MSE.jl")



spectral_grid = SpectralGrid(trunc=31, nlayers=4)


model1 = PrimitiveWetModel(spectral_grid)
model2 = PrimitiveWetModel(spectral_grid)

simulation1 = initialize!(model1)
simulation2 = initialize!(model2)






noise = 0.001 .* (
    randn(Float32, size(simulation1.variables.prognostic.temperature)) .+
    im * randn(Float32, size(simulation1.variables.prognostic.temperature))
)

simulation1.variables.prognostic.temperature .+= noise
simulation2.variables.prognostic.temperature .+= noise


run!(simulation1, period=Hour(1))
run!(simulation2, period=Hour(1))


temp = simulation1.variables.grid.temperature[:, 4] - simulation2.variables.grid.temperature[:, 4]
heatmap(temp, title="Temperature difference [K] at layer 4")

#loss_mse = MSE(simulation1.variables.grid.temperature, simulation2.variables.grid.temperature)