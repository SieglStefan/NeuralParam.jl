### Script for calculating the mean and standard-deviation for global temperature fields for zscore-transformation for NN inputs

using SpeedyWeather 
using Statistics, Random
using CairoMakie

# Include function for perturbing temperature fields
include(joinpath(@__DIR__, "..", "src", "utils", "perturb_vars.jl"))

# Set seed for reproducability


# Define parameters for simulation
const TRUNC = 31            # spectral grid truncation
const NLAYERS = 4           # number of vertical layers

const TSPINUP = 24          # time of spinup after IC pertubation in hours
const N_GAP = 100            # number of timesteps done between sampling

const N_IC = 1             # number of sampled ICs
const N_STEPS = 10         # number sampled steps after spinup per IC

const AMP = 10.              # amplitude of temperature pertubation



# Define spectral grid and model
spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
model = PrimitiveWetModel(spectral_grid)

# Extract timestepping
(; Δt_sec) = model.time_stepping



# Declare temperature fields container
Ts = Float32[]



# Main loop: Define a simulation, perturb temperature, run spinup
for i in 1:N_IC

    # Create simulation
    sim = initialize!(model)
    
    # Perturbate temperature field
    perturb_grid_temp!(sim; amp=AMP)

    display(heatmap(sim.variables.grid.temperature[:,2], title="perturbed temperature"))

    # Spinup model
    run!(sim, period=Hour(TSPINUP))

    display(heatmap(sim.variables.grid.temperature[:,2], title="spinup-ed perturbed temp"))


    # Initialize simulation and do a first step
    initialize!(sim; period=Second((N_STEPS*N_GAP + 1)*Δt_sec))
    SpeedyWeather.first_timesteps!(sim)

    # Propagate the simulation and sample temperature fields
    for j in 1:N_STEPS

        # Propagate simulation N_GAP steps
        for k in 1:N_GAP
            SpeedyWeather.later_timestep!(sim)
        end


        display(heatmap(sim.variables.grid.temperature[:,2], title="step $j"))
        
    end

end
