### Script for calculating the mean and standard-deviation for global temperature fields for zscore-transformation for NN inputs

using SpeedyWeather 
using Statistics, Random
using CairoMakie

# Include function for perturbing temperature fields
include(joinpath(@__DIR__, "..", "src", "utils", "perturb_vars.jl"))

# Define parameters for simulation
const TRUNC = 31            # spectral grid truncation
const NLAYERS = 4           # number of vertical layers

const TSPINUP = 24*1        # time of spinup after IC pertubation in hours
const N_GAP = 10            # number of timesteps done between sampling

const N_IC = 1             # number of sampled ICs
const N_STEPS = 100         # number sampled steps after spinup per IC

const AMP = 1.              # amplitude of temperature pertubation



# Define spectral grid and model
spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
model1 = PrimitiveWetModel(spectral_grid)
model2 = PrimitiveWetModel(spectral_grid)
model3 = PrimitiveWetModel(spectral_grid)

# Extract timestepping
(; Δt, Δt_sec) = model1.time_stepping

# Print information
println("Time after random ICs (days): ", (TSPINUP + Δt_sec*N_STEPS / 3600) / 24)
println("Total number of temperature fields: ", N_IC * N_STEPS)
println("Time between sampling (hours): ", N_GAP * Δt_sec / 3600)

# Declare temperature fields container
Ts1 = Float32[]
Ts2 = Float32[]



# Main loop: Define a simulation, perturb temperature, run spinup
for i in 1:N_IC

    # Initialize model and simulation (without that, variables.grid is not updated and therefore empty)
    sim1 = initialize!(model1)
    sim2 = initialize!(model2)
    sim3 = initialize!(model3)
    #initialize!(sim2)
    
    # Perturbate temperature field
    #perturb_grid_temp!(sim; amp=AMPL)

    # Run model
    #run!(sim1, period=Hour(100*Δt_sec))

    initialize!(sim2; period=Second(10*Δt_sec*100))
    SpeedyWeather.first_timesteps!(sim2)
    
    run!(sim1, period=Second(Δt_sec))
    run!(sim3, period=Second(10*100*Δt_sec+Δt_sec))

    

    # Propagate a certain IC and sample temperature fields
    for j in 1:100


        for i in 1:10
            SpeedyWeather.later_timestep!(sim2)
            # = timestep!(variables, 2Δt, model)               
            # = timestep!(clock, Δt_millisec) 
        end
            append!(Ts2, vec(sim2.variables.grid.temperature))

        # Propagate simulation and store temperatures
        run!(sim1, period=Second(10*Δt_sec))
        append!(Ts1, vec(sim1.variables.grid.temperature))

    end



    display(CairoMakie.heatmap(sim1.variables.grid.temperature[:,4], title="run!"))
    display(CairoMakie.heatmap(sim3.variables.grid.temperature[:,4], title="control"))
    display(CairoMakie.heatmap(sim2.variables.grid.temperature[:,4], title="timestep!"))
    
end


