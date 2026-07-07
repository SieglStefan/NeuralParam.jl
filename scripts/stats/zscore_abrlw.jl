### Script for calculating zscore stats for a AnalyticBandRadiation (LW) parameterization using 
###     standard values for CO2 concentration and ocean-/land-emissivity (lines 56:58)
###
### Objective: Calculate global mean and std for:
### - input variables:
###     - temperature for every layer k
###     - log10 humidity for every layer k
###     - surface pressure
###     - sea surface temperature
###     - land surface temperature
###     - land fraction
### - output variables
###     - temperature tendencies for every layer k
###     - flux tendencies
###
### Additionally:
###     - displays plots of heatmaps of perturbation before and after spinup for checking
###     - stores and displays a plot of the vertical temperature profile at stats/plots
###     - stores meta data of creation of stats at stats/info
###     - stores histograms of normalized input and output variables at stats/plots



### Load packages and code
using Revise
using NeuralParam
using SpeedyWeather, AnalyticBandRadiation
const SpeedyExt = Base.get_extension(AnalyticBandRadiation,
                                     :AnalyticBandRadiationSpeedyWeatherExt)
using Statistics, Random, Dates
using JLD2
using CairoMakie



### Define stats parameters
# General
NAME = ""             # name of statistics
CREATED = now()       # date of creation
SEED = 1234           # seed used    

# Grid
TRUNC = 31            # truncation of spectral grid
NLAYERS = 8           # number of vertical layers

# Sampling
T_SPINUP = 30         # spinup time in days
N_IC = 3              # number of initial conditions
SIM_TIME = 365        # sampling time in days
SAMPLE_GAP = 3.65     # days between sampling

# Perturbation
AMP_T = 2f0           # temperature perturbation amplitude (additive)
AMP_Q = 0.2f0         # humidity perturbation amplitude (multiplicative)

# Used standard values
CO2 = 280f0           # default CO2 value 
OCEAN_EM = 1f0        # default ocean emissivity value (SW does not propagate yet)
LAND_EM = 1f0         # default land emissivity value (SW does not propagate yet)


# Folder for stats files (.jld2, .png and .toml)
foldername = "zscore_abrlw_L$(NLAYERS)$(NAME)"
folderpath = joinpath(@__DIR__, "..", "..", "data", "stats", foldername)
mkdir(folderpath)



### Prepare simulation
# Set seed for reproducability
Random.seed!(SEED)

# Define spectral grid and model
spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
LW_SCHEME = SpeedyExt.SpeedyAnalyticBandLongwave(spectral_grid)       # used parameterization scheme
model = PrimitiveWetModel(spectral_grid; longwave_radiation = LW_SCHEME)

# Create template simulation
sim_temp = initialize!(model)

# Extract timestepping
(; Δt_sec) = model.time_stepping


# Calulate number of timesteps for sampling
n_steps_total = round(Int, SIM_TIME *3600 *24 / Δt_sec) + 1
n_gap = round(Int, SAMPLE_GAP *3600 *24 / Δt_sec)


# Declare input field container
T_layers = [Float32[] for _ in 1:NLAYERS]
q_layers = [Float32[] for _ in 1:NLAYERS]
p = Float32[]
    #co2 = Float32[]        not needed, because standard value is always used
sst = Float32[]
lst = Float32[]
    #lf = Float32[]         not needed, because land fraction is already in [0,1]
    #ocean_em = Float32[]   not needed, because standard value is always used
    #land_em = Float32[]    not needed, because standard value is always used

# Declare output field container
dT_layers = [Float32[] for _ in 1:NLAYERS]
olw = Float32[]
slwd = Float32[] 



# Print information
println("--------------------------------------------------------------------")
println("Total number of global samples per IC: ", n_steps_total ÷ n_gap)
println("Time between samples (days): ", n_gap * Δt_sec /3600 /24)
println("--------------------------------------------------------------------")



### Main loop: Define a simulation, perturb temperature and humidity, run spinup
for i in 1:N_IC

    # Create simulation
    sim = deepcopy(sim_temp)
    
    # Perturbate temperature and humidity fields
    perturb_grid_field!(sim, :temperature; fac_add=AMP_T)
    perturb_grid_field!(sim, :humidity; fac_mult=AMP_Q, zeromin=true)


    # Plot heatmap if first ic for visualization of perturbation before spinup
    titles = ["TOA", "Between", "Surface"]
    if i == 1
        temp_vec = [sim.variables.grid.temperature[:,k] for k in [1, Int(NLAYERS/2), NLAYERS]]
        hum_vec = [log10.(sim.variables.grid.humidity[:,k] .+ 1f-9) for k in [1, Int(NLAYERS/2), NLAYERS]]
        
        display(plot_heatmaps(temp_vec; titles, suptitle = "Temperature before Spinup", coastlines=false))
        display(plot_heatmaps(hum_vec; titles, suptitle = "Humidity before Spinup", coastlines=false))
    end


    # Spinup model
    run!(sim, period=Day(T_SPINUP))


    # Plot heatmap if first ic for visualization of perturbation after spinup
    titles = ["TOA", "Between", "Surface"]
    if i == 1
        temp_vec = [sim.variables.grid.temperature[:,k] for k in [1, Int(NLAYERS/2), NLAYERS]]
        hum_vec = [log10.(sim.variables.grid.humidity[:,k] .+ 1f-9) for k in [1, Int(NLAYERS/2), NLAYERS]]
        
        display(plot_heatmaps(temp_vec; titles, suptitle = "Temperature after Spinup", coastlines=false))
        display(plot_heatmaps(hum_vec; titles, suptitle = "Humidity after Spinup", coastlines=false))
    end


    # Initialize simulation and do a first step
    initialize!(sim; steps=n_steps_total)
    SpeedyWeather.first_timesteps!(sim)


    # Propagate the simulation and sample temperature fields
    for step in 1:n_steps_total

        # Do a timestep
        SpeedyWeather.later_timestep!(sim)
        
        # Store stats after n_gap steps
        if step % n_gap == 0
            # Shortcut variables
            vars = sim.variables


            ### Collect input fields
            for k in 1:NLAYERS
                append!(T_layers[k], vec(vars.grid.temperature_prev[:, k]))
                append!(q_layers[k], vec(log10.(vars.grid.humidity_prev[:, k] .+ 1f-9)))
            end

            append!(p,  vec(vars.grid.pressure_prev))
            append!(sst, vec(vars.prognostic.ocean.sea_surface_temperature))
            append!(lst, vec(vars.prognostic.land.soil_temperature[:, 1]))


            ### Collect output fields by explicitely calling the parameterization for tendencies
            for ij in axes(vars.grid.temperature_prev, 1)

                T_col = @view vars.grid.temperature_prev[ij, :]
                q_col = @view vars.grid.humidity_prev[ij, :]
                Φ_col = @view vars.grid.geopotential[ij, :]
                p_col = vars.grid.pressure_prev[ij]

                # Create surface state for ABR call
                surface = SurfaceState{Float32}(
                    sea_surface_temperature  = vars.prognostic.ocean.sea_surface_temperature[ij],
                    land_surface_temperature = vars.prognostic.land.soil_temperature[ij, 1],
                    land_fraction            = model.land_sea_mask.mask[ij],
                    ocean_emissivity         = OCEAN_EM,
                    land_emissivity          = LAND_EM,
                )

                # Create profile for ABR call
                profile = AtmosphereProfile(
                    temperature      = T_col,
                    humidity         = q_col,
                    geopotential     = Φ_col,
                    surface_pressure = p_col,
                    CO₂              = CO2,
                )

                # Solve for temperature tendencies and fluxes
                dTdt = zeros(Float32, NLAYERS)
                diag = LongwaveDiagnostics{Float32}()

                solve_longwave!(
                    dTdt,
                    diag,
                    LW_SCHEME.scheme,
                    profile,
                    SpeedyExt._speedy_column_geometry(model),
                    surface,
                    SpeedyExt._speedy_physical_constants(model),
                )

                # Store values
                for k in 1:NLAYERS
                    push!(dT_layers[k], Float32(dTdt[k]))
                end

                append!(olw, diag.outgoing_longwave)
                append!(slwd, diag.surface_longwave_down)
            end
        end
    end

    println("\tIC $i finished")
end


### Calculate stats
# Calculate mean and std of input fields
T_mean = Float32[mean(T_layers[k]) for k in 1:NLAYERS]
T_std  = Float32[std(T_layers[k])  for k in 1:NLAYERS]

q_mean = Float32[mean(q_layers[k]) for k in 1:NLAYERS]
q_std  = Float32[std(q_layers[k])  for k in 1:NLAYERS]

p_mean  = Float32[mean(p)]
p_std   = Float32[std(p)]

co2_mean = CO2
co2_std = 1f0

sst_mean = Float32[mean(sst)]
sst_std  = Float32[std(sst)]

lst_mean = Float32[mean(lst)]
lst_std  = Float32[std(lst)]

lf_mean  = 0f0
lf_std   = 1f0

ocean_em_mean = OCEAN_EM
ocean_em_std = 1f0

land_em_mean = LAND_EM
land_em_std = 1f0

# Caclulate mean and std of output fields
dT_mean = Float32[mean(dT_layers[k]) for k in 1:NLAYERS]
dT_std  = Float32[std(dT_layers[k])  for k in 1:NLAYERS]

olw_mean  = Float32[mean(olw)]
olw_std   = Float32[std(olw)]

slwd_mean  = Float32[mean(slwd)]
slwd_std   = Float32[std(slwd)]


# Combine to input mean and std
input_mean = vcat(T_mean, q_mean, p_mean, co2_mean, sst_mean, lst_mean, lf_mean, ocean_em_mean, land_em_mean)
input_std  = vcat(T_std, q_std, p_std, co2_std, sst_std, lst_std, lf_std, ocean_em_std, land_em_std)

# Combine to output mean and std
output_mean = vcat(dT_mean, olw_mean, slwd_mean)
output_std  = vcat(dT_std, olw_std, slwd_std)



### Store statistics
file = "stats.jld2"
filepath = joinpath(folderpath, file)

JLD2.jldsave(filepath;
    input_mean,
    input_std,
    output_mean,
    output_std,
)



### Plot vector results
layers = 1:NLAYERS

fig = Figure()

# Temperature plot
ax_T = Axis(
    fig[1, 1],
    xlabel = "Temperature (K)",
    ylabel = "Layer",
    title = "Vertical Temperature Profile",
    yticks = layers,
    yreversed = true,
)

lines!(ax_T, T_mean, layers)
scatter!(ax_T, T_mean, layers)
errorbars!(ax_T, T_mean, layers, T_std; direction = :x, whiskerwidth = 8)

# Humidity plot
ax_q = Axis(
    fig[1, 2],
    xlabel = "Log10 Specific Humidity",
    ylabel = "Layer",
    title = "Vertical Humidity Profile",
    yticks = layers,
    yreversed = true,
)

lines!(ax_q, q_mean, layers)
scatter!(ax_q, q_mean, layers)
errorbars!(ax_q, q_mean, layers, q_std; direction = :x, whiskerwidth = 8)

# Display plot
display(fig)

# Store plot
file = "temp_humid_vertical_profile.png"
filepath = joinpath(folderpath, file)
CairoMakie.save(filepath, fig)



### Print single results and dT (without plots):
println("")
println("Scalar Results:")
println("\tInputs:")
println("\t\t surface pressure mean: $p_mean")
println("\t\t surface pressure std: $p_std")
println("")
println("\t\t sea surface temperature mean: $sst_mean")
println("\t\t sea surface temperature std: $sst_std")
println("")
println("\t\t land surface temperature mean: $lst_mean")
println("\t\t land surface temperature std: $lst_std")
println("")
println("\tOutputs:")
println("\t\t Temperature tendencies mean: $dT_mean")
println("\t\t Temperature tendencies std: $dT_std")
println("")
println("\t\t outgoing LW mean: $olw_mean")
println("\t\t outgoing LW std: $olw_std")
println("")
println("\t\t Surface LW down mean: $slwd_mean")
println("\t\t Surface LW down std: $slwd_std")
println("")



### Create and store meta data .toml file
write_info(;
    path = folderpath,
    file = "meta.toml",

    name =          NAME,
    created =       CREATED,
    seed =          SEED,
    julia =         string(VERSION),

    inputs =        ["Temperature profile, Humidity profile, surface pressure, CO2 concentration, Sea surface temperature, Land surface temperature, Land fraction, Ocean emissivity, Land Emissivity"],
    outputs =       ["Tempertaure profile tendencies, Outgoing longwave, Surface longwave down"],
    
    defaults =      ["CO2 concentration, Ocean emissivity, Land Emissivity"],
    defaults_vals = [CO2, OCEAN_EM, LAND_EM],

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
    amp_q =         AMP_Q,
)



### Create histogram plots for validation
# Input fields
histo_T = plot_histograms(
    zscore.(T_layers, T_mean, T_std), 
    ["Layer $k" for k in 1:NLAYERS]; 
    suptitle = "Normalized Temperature Histograms", 
    ncols = NLAYERS÷2)
histo_q = plot_histograms(
    zscore.(q_layers, q_mean, q_std), 
    ["Layer $k" for k in 1:NLAYERS]; 
    suptitle = "Normalized Log10 Humidity Histograms", 
    ncols = NLAYERS÷2)
hist_scalar_in = plot_histograms(
    [zscore(p, p_mean, p_std), zscore(sst, sst_mean, sst_std), zscore(lst, lst_mean, lst_std)], 
    ["Surface Pressure", "Sea Surface Temperature", "Land Surface Temperature", "Land Fraction"],
    suptitle = "Scalar Input Histograms", 
    ncols = 4)

# Output fields
hist_dT = plot_histograms(
    zscore.(dT_layers, dT_mean, dT_std), 
    ["Layer $k" for k in 1:NLAYERS]; 
    suptitle = "Normalized Temperature Tendencies Histograms", 
    ncols = NLAYERS÷2)
hist_fluxes = plot_histograms(
    [zscore(olw, olw_mean, olw_std), zscore(slwd, slwd_mean, slwd_std)], 
    ["Outgoing Longwave", "Surface Longwave Down"],
    suptitle = "Normalized Flux Histograms",
    ncols = 2)


# Save Histograms
histopath = joinpath(folderpath, "histograms")
mkpath(histopath)

CairoMakie.save(joinpath(histopath, "input_T.png"), histo_T)
CairoMakie.save(joinpath(histopath, "input_q.png"), histo_q)
CairoMakie.save(joinpath(histopath, "input_scalar.png"), hist_scalar_in)

CairoMakie.save(joinpath(histopath, "output_dT.png"), hist_dT)
CairoMakie.save(joinpath(histopath, "output_fluxes.png"), hist_fluxes)