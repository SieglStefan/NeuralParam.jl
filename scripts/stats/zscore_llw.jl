### Script for calculating zscore stats for a NeuralLinearLongwave parameterization
###
### Objective: Calculate global mean and std for:
### - input variables:
###     - temperature per layer k
### - output variables:
###     - nothing: is scaled with ConstLinearLW parameters
###
### Additionally:
###     - displays plots of heatmaps of perturbation before and after spinup for checking
###     - stores and displays a plot of the vertical temperature profile at stats/plots
###     - stores meta data of creation of stats at stats/info
###     - stores histograms of normalized input variables at stats/histo



### Load packages
using Revise
using NeuralParam
using SpeedyWeather
using Statistics, Random, Dates
using JLD2
using CairoMakie



### Define parameters for sampling
# General
NAME = ""             # name of statistics
CREATED = now()       # date of creation
SEED = 1234           # seed used    

# Grid
TRUNC = 31            # truncation of spectral grid
NLAYERS = 8           # number of vertical layers

# Sampling
T_SPINUP = 30         # spinup time in days
N_IC = 5              # number of initial conditions
SIM_TIME = 365        # sampling time in days
SAMPLE_GAP = 3.65     # days between sampling

# Perturbation
AMP_T = 2f0           # temperature perturbation amplitude
AMP_Q = 0.2f0         # humidity perturbation amplitude (multiplicative)


# Folder for stats files (.jld2, .png and .toml)
foldername = "zscore_llw_L$(NLAYERS)$(NAME)"
folderpath = joinpath(@__DIR__, "..", "..", "data", "stats", foldername)
mkdir(folderpath)



### Prepare simulation
# Set seed for reproducability
Random.seed!(SEED)

# Define spectral grid and model
spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
LW_SCHEME = OneBandLongwave(spectral_grid)        # used parameterization scheme
model = PrimitiveWetModel(spectral_grid; longwave_radiation = LW_SCHEME)

# Create template simulation
sim_temp = initialize!(model)

# Extract timestepping
(; Δt_sec) = model.time_stepping


# Calulate number of timesteps for sampling
n_steps_total = round(Int, SIM_TIME *3600 *24 / Δt_sec) + 1
n_gap = round(Int, SAMPLE_GAP *3600 *24 / Δt_sec)


# Declare temperature field container
T_layers = [Float32[] for _ in 1:NLAYERS]



# Print information
println("--------------------------------------------------------------------")
println("Total number of global samples per IC: ", n_steps_total ÷ n_gap)
println("Time between samples (days): ", n_gap * Δt_sec /3600 /24)
println("--------------------------------------------------------------------")



### Main loop: Define a simulation, perturb temperature, run spinup
for i in 1:N_IC

    # Create simulation
    sim = deepcopy(sim_temp)
    
    # Perturbate temperature field
    perturb_grid_field!(sim, :temperature; fac_add=AMP_T)


    # Plot heatmap if first ic for visualization of perturbation before spinup
    titles = ["TOA", "Between", "Surface"]
    if i == 1
        temp_vec = [sim.variables.grid.temperature[:,k] for k in [1, Int(NLAYERS/2), NLAYERS]]
        
        display(plot_heatmaps(temp_vec; titles, suptitle = "Temperature before Spinup", coastlines=false))
    end


    # Spinup model
    run!(sim, period=Day(T_SPINUP))


    # Plot heatmap if first ic for visualization of perturbation after spinup
    titles = ["TOA", "Between", "Surface"]
    if i == 1
        temp_vec = [sim.variables.grid.temperature[:,k] for k in [1, Int(NLAYERS/2), NLAYERS]]

        display(plot_heatmaps(temp_vec; titles, suptitle = "Temperature after Spinup", coastlines=false))
    end


    # Initialize simulation and do a first step
    initialize!(sim; steps=n_steps_total)
    SpeedyWeather.first_timesteps!(sim)


    # Propagate the simulation and sample temperature fields
    for step in 1:n_steps_total

        # Do a timestep
        SpeedyWeather.later_timestep!(sim)

        # Store input temperatures after n_gap steps
        if step % n_gap == 0
            for k in 1:NLAYERS
                append!(T_layers[k], vec(sim.variables.grid.temperature_prev[:, k]))
            end
        end
    end

    # Print information
    println("\t\tIC Nr. $i finished!")
end



### Calculate T_mean and T_std 
T_mean = Float32[mean(T_layers[k]) for k in 1:NLAYERS]
T_std  = Float32[std(T_layers[k])  for k in 1:NLAYERS]

# Combine to input mean and std
input_mean = T_mean
input_std = T_std

# Output mean and std are not used, but need to be defined
output_mean = 0f0
output_std = 0f0



### Store statistics
file = "stats.jld2"
filepath = joinpath(folderpath, file)

JLD2.jldsave(filepath;
    input_mean,
    input_std,
    output_mean,
    output_std,
)



### Plot results
layers = 1:NLAYERS

fig = Figure()

ax = Axis(
    fig[1, 1],
    xlabel = "Temperature (K)",
    ylabel = "Layer",
    title = "Vertical Temperature Profile",
    yticks = layers,
    yreversed = true
)

# T_mean as lines and points
lines!(ax, T_mean, layers)
scatter!(ax, T_mean, layers)

# Horizontal error bars
errorbars!(ax, T_mean, layers, T_std; direction = :x)

# Display plot
display(fig)

# Store plot
# Store plot
file = "temp_vertical_profile.png"
filepath = joinpath(folderpath, file)
CairoMakie.save(filepath, fig)



### Create and store meta data .toml file
write_info(;
    path = folderpath,
    file = "meta.toml",

    name =          NAME,
    created =       CREATED,
    seed =          SEED,
    julia =         string(VERSION),

    inputs =        ["temperature"],
    outputs =       ["none: scaled via ConstLinearLW parameters"],

    lw_scheme =     string(nameof(typeof(LW_SCHEME))),   

    trunc =         TRUNC,
    nlayers =       NLAYERS,

    t_spinup =      T_SPINUP,
    n_ic =          N_IC,
    sim_time =      SIM_TIME,
    sample_gap =    SAMPLE_GAP,

    n_stats =       n_steps_total ÷ n_gap,
    gap_real =      n_gap * Δt_sec /3600 /24,

    amp_t =         AMP_T,
)

write_info(;
    path = folderpath,
    file = "meta.toml",

    provenance = (;
        name    = NAME,
        created = CREATED,
        seed    = SEED,
        julia   = string(VERSION),
    ),

    io = (;
        inputs =        ["temperature"],
        outputs =       ["none: scaled via ConstLinearLW parameters"],
    ),

    scheme = (;
        lw_scheme = string(nameof(typeof(LW_SCHEME))),
    ),

    grid = (;
        trunc   = TRUNC,
        nlayers = NLAYERS,
    ),

    sampling = (;
        t_spinup   = T_SPINUP,
        n_ic       = N_IC,
        sim_time   = SIM_TIME,
        sample_gap = SAMPLE_GAP,
        n_stats    = n_steps_total ÷ n_gap,
        gap_real   = n_gap * Δt_sec / 3600 / 24,
    ),

    perturbation = (;
        amp_t = AMP_T,
        amp_q = AMP_Q,
    ),
)



### Create histogram plots for validation
histo_T = plot_histograms(
    zscore.(T_layers, T_mean, T_std), 
    ["Layer $k" for k in 1:NLAYERS];
    suptitle = "Normalized Temperature Histograms",
    ncols = NLAYERS÷2)

# Save histograms
histopath = joinpath(folderpath, "histograms")
mkpath(histopath)

CairoMakie.save(joinpath(histopath, "input_T.png"), histo_T)