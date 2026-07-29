### Script for calculating zscore stats for a NeuralLinearLongwave parameterization
###
### Objective: Calculate global mean and std for:
### - input variables:
###     - temperature per layer k
### - output variables:
###     - nothing: output is scaled with ConstLinearLW parameters
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
using CairoMakie



### Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)



### Define parameters for sampling
# General
NAME        = "default"             # name of zscore
SEED        = 21                    # seed used    

# Model and scheme
MODEL       = PrimitiveWetModel                                                                 # used model
LW_SCHEME   = OneBandLongwave(SG; transmissivity = FriersonLongwaveTransmissivity(SG))          # used LW scheme

# Sampling
T_SPINUP    = Day(30)                    # spinup time in days
START_DATE  = DateTime(2000, 1, 1)      # start date of simulation
N_IC        = 1                     # number of initial conditions
SIM_TIME    = 20                   # sampling time in days
SAMPLE_GAP  = 3.65                  # days between sampling

# Perturbation
FAC_PERT_T  = 2f0                   # temperature perturbation amplitude
FAC_PERT_Q  = 0.2f0                 # humidity perturbation amplitude (multiplicative)



### Prepare simulation
# Set seed for reproducability
Random.seed!(SEED)

# Create model and initialize simulation
model = MODEL(SG; longwave_radiation = LW_SCHEME)
sim_temp = initialize!(model)

# Set starting time for spinup
clock_start = START_DATE - T_SPINUP
SpeedyWeather.set!(sim_temp.variables.prognostic.clock; time = clock_start, start = clock_start)

# Extract timestepping
(; Δt_sec) = model.time_stepping


# Calulate number of timesteps for sampling
n_steps_total = round(Int, SIM_TIME *3600 *24 /Δt_sec) + 1
n_gap = round(Int, SAMPLE_GAP *3600 *24 /Δt_sec)

# Print information
@info "Total number of global samples per IC: $(n_steps_total ÷ n_gap)"
@info "Time between samples (days): $(n_gap * Δt_sec /3600 /24)"


# Declare temperature field container
T_layers = [Float32[] for _ in 1:NLAYERS]



### Main loop: Define a simulation, perturb temperature, run spinup
for i in 1:N_IC

    # Create simulation
    sim = deepcopy(sim_temp)
    
    # Perturbate temperature and humidity fields
    perturb_grid_field!(sim, :temperature; fac_add=FAC_PERT_T)
    perturb_grid_field!(sim, :humidity; fac_mult=FAC_PERT_Q, zeromin=true)


    # Plot heatmap if first ic for visualization of perturbation before spinup
    titles = ["TOA", "Between", "Surface"]
    if i == 1
        temp_vec = [sim.variables.grid.temperature[:,k] for k in [1, Int(NLAYERS÷2), NLAYERS]]
        hum_vec = [log10.(sim.variables.grid.humidity[:,k] .+ 1f-9) for k in [1, Int(NLAYERS÷2), NLAYERS]]
        
        display(plot_heatmaps(temp_vec; titles, suptitle = "Temperature before Spinup", coastlines=false))
        display(plot_heatmaps(hum_vec; titles, suptitle = "Log10 Humidity before Spinup", coastlines=false))
    end


    # Spinup model
    run!(sim, period=T_SPINUP)


    # Plot heatmap if first ic for visualization of perturbation after spinup
    titles = ["TOA", "Between", "Surface"]
    if i == 1
        temp_vec = [sim.variables.grid.temperature[:,k] for k in [1, Int(NLAYERS÷2), NLAYERS]]
        hum_vec = [log10.(sim.variables.grid.humidity[:,k] .+ 1f-9) for k in [1, Int(NLAYERS÷2), NLAYERS]]
        
        display(plot_heatmaps(temp_vec; titles, suptitle = "Temperature after Spinup", coastlines=false))
        display(plot_heatmaps(hum_vec; titles, suptitle = "Log10 Humidity after Spinup", coastlines=false))
    end


    # Initialize simulation and do a first step
    initialize!(sim; steps=n_steps_total)
    SpeedyWeather.first_timesteps!(sim)


    # Propagate the simulation and sample fields
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
output_std = 1f0



### Store statistics
# Create folder
foldername = "zscore_llw_$(NAME)"
folderpath = prepare_out_dir(stats_dir(), foldername)

# Save stats
NeuralParam.save(
    (input_mean=input_mean, input_std=input_std, output_mean=output_mean, output_std=output_std); 
    path=folderpath, file="stats.jld2"
)
@info "Statistics $(foldername) stored at $(folderpath)!"


# Create and store info.toml file
write_info(;
    path = folderpath,
    file = "info.toml",

    general = (;
        name    = NAME,
        created = now(),
        seed    = SEED,
        julia   = string(VERSION),
        sw_vers = string(pkgversion(SpeedyWeather)),
    ),

    io = (;
        inputs  = ["temperature"],
        outputs = ["none: scaled via ConstLinearLW parameters"],    
    ),

    stats = (;
        input = (;
            order   = ["T"],
            lengths = [NLAYERS],
            T_mean  = T_mean,       T_std = T_std,
        ),

        output = (;
            order   = [],
            lengths = [],
        ),
    ),

    grid = (;
        trunc   = TRUNC,
        nlayers = NLAYERS,
        grid_type   = string(nameof(SG.Grid)),
    ),

    model_type = (;
        model       = nameof(MODEL),
        lw_scheme   = string(nameof(typeof(LW_SCHEME))),   
    ),

    sampling = (;
        t_spinup    = string(T_SPINUP),
        start_date  = string(START_DATE),
        n_ic        = N_IC,
        sim_time    = SIM_TIME,
        sample_gap  = SAMPLE_GAP,
        n_stats     = n_steps_total ÷ n_gap,
        gap_real    = n_gap * Δt_sec /3600 /24,
    ),

    perturbation = (;
        fac_pert_t = FAC_PERT_T,
        fac_pert_q = FAC_PERT_Q,
    ),
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
file = "temp_vertical_profile.png"
filepath = joinpath(folderpath, file)
CairoMakie.save(filepath, fig)



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