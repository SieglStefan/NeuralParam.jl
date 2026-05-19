using SpeedyWeather



spectral_grid = SpectralGrid(trunc=31, nlayers=4)
model = PrimitiveWetModel(spectral_grid)
sim = initialize!(model)


sim1 = deepcopy(sim)
sim2 = deepcopy(sim)


run!(sim1)
run!(sim2)

display(CairoMakie.heatmap(sim1.variables.grid.temperature[:,4], title="before"))
display(CairoMakie.heatmap(sim2.variables.grid.temperature[:,4], title="before"))

for i in 1:1000
    initialize!(sim)
end

run!(sim1)
run!(sim2)


display(CairoMakie.heatmap(sim1.variables.grid.temperature[:,4], title="after"))
display(CairoMakie.heatmap(sim2.variables.grid.temperature[:,4], title="after"))