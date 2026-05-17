using SpeedyWeather, CairoMakie




spectral_grid = SpectralGrid(trunc=31, nlayers=8)
model1 = PrimitiveWetModel(spectral_grid)
model2 = PrimitiveWetModel(spectral_grid)
sim1 = initialize!(model1)
sim2 = initialize!(model2)



#model1.time_stepping.start_with_euler = false
#model2.time_stepping.start_with_euler = true

#model1.time_stepping.continue_with_leapfrog = false
#model2.time_stepping.continue_with_leapfrog = true




run!(sim1, period=Day(1))
run!(sim2, period=Day(1))


model1.time_stepping.start_with_euler = true
model2.time_stepping.start_with_euler = false

model1.time_stepping.continue_with_leapfrog = true
model2.time_stepping.continue_with_leapfrog = false


run!(sim1, period=Day(1))
run!(sim2, period=Day(1))


#Δt_sec = model1.time_stepping.Δt_sec
#Δt = model1.time_stepping.Δt
#Δt_millisec = model1.time_stepping.Δt_millisec


sim1.variables.grid == sim2.variables.grid




