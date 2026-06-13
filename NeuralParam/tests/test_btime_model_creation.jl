using SpeedyWeather, BenchmarkTools

spectral_grid = SpectralGrid(;trunc=31, nlayers=4)
model0 = PrimitiveWetModel(; spectral_grid)


@btime model1 = PrimitiveWetModel(; spectral_grid)
@btime sim1 = initialize!(model0)

model1 = PrimitiveWetModel(; spectral_grid = $spectral_grid)
sim1 = initialize!($model0)

@btime deepcopy($sim1)