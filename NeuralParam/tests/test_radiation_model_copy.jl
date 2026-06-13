using SpeedyWeather


my_radiation = OneBandLongwaveRadiativeTransfer()

spectral_grid = SpectralGrid(trunc=31, nlayers=4)
model = PrimitiveWetModel(spectral_grid, longwave_radiation=my_radiation)

my_radiation === model.longwave_radiation