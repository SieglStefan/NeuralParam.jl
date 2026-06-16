### Experiment 1.1: LinearLongwave calibration
    # Goal: 
    # Problems:
    # Results:
    # Conclusion:


using Revise
using NeuralParam

using SpeedyWeather
using Dates

println("Start time: $(now())")
t0 = now()

# Standard grid
spectral_grid = SpectralGrid(trunc=31, nlayers=1)
model = PrimitiveWetModel(spectral_grid)

#const N_UPDATES = 5

#dt = model.time_stepping.Δt_sec
#tges = dt * N_UPDATES /(3600*24)

#println("-------------------------")
#println("1 timestep = $dt seconds")
#println(" -> whole update time: $tges days")
#println("-------------------------")


radiation_llw = LinearLongwave(a = -1f-9, b = 1f-5, sc_a = 1f-8, sc_b = 1f-6)

L, P, G = run_calibration!( radiation_llw, 
                            spectral_grid;
                            eta0 = 1f-1,           
                            n_ic = 2,              
                            n_traj = 10,
                            n_epochs = 5,
                            n_gap = 10,
                            n_steps = 10)

                            
a = first.(P)
b = last.(P)

plot_calibration(L, abs.(a), abs.(b))


t1 = now()
runtime_min = Dates.value(t1 - t0) / (1000 * 60)

println("End time:   $(t1)")
println("Runtime:    $(runtime_min) minutes")

a_end = a[end]
b_end = b[end]

T_eq = -b_end / a_end

println("T_eq: $T_eq")