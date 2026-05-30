### Script for calculating the mean and standard-deviation for global temperature fields for zscore-transformation for NN inputs

using SpeedyWeather 
using Statistics, Random
using CairoMakie

# Include function for perturbing temperature fields
include(joinpath(@__DIR__, "..", "src", "utils", "perturb_vars.jl"))

# Set seed for reproducability
Random.seed!(1234)

# Define parameters for simulation
const TRUNC = 31            # spectral grid truncation
const NLAYERS = 4           # number of vertical layers

const TSPINUP = 24          # time of spinup after IC pertubation in hours
const N_GAP = 10            # number of timesteps done between sampling

const N_IC = 10             # number of sampled ICs
const N_STEPS = 100         # number sampled steps after spinup per IC

const AMP = 1.              # amplitude of temperature pertubation



# Define spectral grid and model
spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
model = PrimitiveWetModel(spectral_grid)

# Extract timestepping
(; Δt_sec) = model.time_stepping

# Print information
println("Total time propagated (days): ", ((TSPINUP + Δt_sec*(N_STEPS*N_GAP+1)) / 3600) / 24)
println("Total number of temperature fields: ", N_IC * N_STEPS)
println("Time between sampling (hours): ", N_GAP * Δt_sec / 3600)

# Declare temperature fields container
Ts = Float32[]



# Main loop: Define a simulation, perturb temperature, run spinup
for i in 1:N_IC

    # Create simulation
    sim = initialize!(model)
    
    # Perturbate temperature field
    perturb_grid_temp!(sim; amp=AMP)

    # Spinup model
    run!(sim, period=Hour(TSPINUP))


    # Initialize simulation and do a first step
    initialize!(sim; period=Second((N_STEPS*N_GAP + 1)*Δt_sec))
    SpeedyWeather.first_timesteps!(sim)

    # Propagate the simulation and sample temperature fields
    for j in 1:N_STEPS

        # Propagate simulation N_GAP steps
        for k in 1:N_GAP
            SpeedyWeather.later_timestep!(sim)
        end

        # Store temperatures
        append!(Ts, vec(sim.variables.grid.temperature))
        
    end

end



# Calculate T_mean and T_std 
T_mean = mean(Ts)
T_std = std(Ts)

# Print
println("T_mean: ", T_mean)
println("T_std: ", T_std)
