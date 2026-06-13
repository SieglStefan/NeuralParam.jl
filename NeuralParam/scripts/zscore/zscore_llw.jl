### Script for calculating mean and std data for zscore-transformation for 
### NN inputs for a NeuralLinearLongwave parameterization
###
### Objective: Calculate global mean and std for:
### - temperature for every layer k



using SpeedyWeather
using Statistics, Random, Dates
using JLD2
using CairoMakie

include(joinpath(@__DIR__, "..", "..", "src", "utils", "data.jl"))

Random.seed!(1234)

const TRUNC = 31
const NLAYERS = 8

const T_SPINUP = 14     # days
const N_IC = 10         # number of initial conditions
const N_STEPS = 50      # samples per IC
const N_GAP = 10        # timesteps between samples
const AMP = 2f0         # temperature perturbation amplitude



# Define spectral grid and model
spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
model = PrimitiveWetModel(spectral_grid)

# Create template simulation
sim_temp = initialize!(model)

# Extract timestepping
(; Δt_sec) = model.time_stepping

# Print information
println("--------------------------------------------------------------------")
println("Total time propagated per IC (days): ", ((T_SPINUP + Δt_sec*(N_STEPS*N_GAP+1)) / 3600) / 24)
println("Time between samples (hours): ", N_GAP * Δt_sec / 3600)
println("Total number of global temperature fields: ", N_IC * N_STEPS)
println("--------------------------------------------------------------------")



# Declare temperature field container
T_layers = [Float32[] for _ in 1:NLAYERS]



# Main loop: Define a simulation, perturb temperature, run spinup
for i in 1:N_IC

    # Create simulation
    sim = deepcopy(sim_temp)
    
    # Perturbate temperature field
    perturb_grid_temp!(sim; amp=AMP)

    # Spinup model
    run!(sim, period=Day(T_SPINUP))


    # Initialize simulation and do a first step
    initialize!(sim; steps=(N_STEPS*N_GAP+1))
    SpeedyWeather.first_timesteps!(sim)

    # Propagate the simulation and sample temperature fields
    for j in 1:N_STEPS

        # Propagate simulation N_GAP steps
        for k in 1:N_GAP
            SpeedyWeather.later_timestep!(sim)
        end

        # Store temperatures
        for l in 1:NLAYERS
            append!(T_layers[l], vec(sim.variables.grid.temperature[:, l]))
        end
    end

    # Print information
    println("\t\tIC Nr. $i finished!")
end



# Calculate T_mean and T_std 
T_mean = Float32[mean(T_layers[k]) for k in 1:NLAYERS]
T_std  = Float32[std(T_layers[k])  for k in 1:NLAYERS]

# Combine to input mean and std
input_mean = T_mean
input_std = T_std



# Store statistics
file = "llw_T$(TRUNC)_L$(NLAYERS).jld2"
path = joinpath(@__DIR__, "..", "..", "data", "zscore", file)

mkpath(dirname(path))

JLD2.jldsave(path;
    input_mean,
    input_std,
)



# Plot results
layers = 1:NLAYERS

fig = Figure()

ax = Axis(
    fig[1, 1],
    xlabel = "Temperature (K)",
    ylabel = "Layer",
    title = "Global mean temperature profile with std",
    yticks = layers,
    yreversed = true
)

# T_mean as lines and points
lines!(ax, T_mean, layers)
scatter!(ax, T_mean, layers)

# Horizontal error bars
errorbars!(ax, T_mean, layers, T_std; direction = :x)

display(fig)