### Experiment 1.1: LinearLongwave calibration
    # Goal: 
    # Problems:
    # Results:
    # Conclusion:



using SpeedyWeather
using Dates

println("Start time: $(now())")
t0 = now()

# Standard grid
spectral_grid = SpectralGrid(trunc=31, nlayers=4)

# Define some initial parameters
radiation_nllw = NeuralLinearLongwave(spectral_grid)



L, P, G = run_training!(radiation_nllw, 
                        spectral_grid;
                        eta0 = 1f-4,          
                        n_ic = 1,              
                        n_updates = 15,
                        n_gap = 10)


plot_loss(L)


t1 = now()

println("End time:   $(t1)")
println("Runtime:    $(t1 - t0)")
