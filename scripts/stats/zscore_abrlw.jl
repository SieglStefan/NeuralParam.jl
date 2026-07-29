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
SpeedyExt = Base.get_extension(AnalyticBandRadiation,
                                     :AnalyticBandRadiationSpeedyWeatherExt)
using Statistics, Random, Dates
using CairoMakie



### Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)



### Define parameters for sampling
# General
NAME        = "default"             # name of statistics
SEED        = 1234                    # seed used    

# Model and scheme
MODEL       = PrimitiveWetModel                                 # used model
LW_SCHEME   = SpeedyExt.SpeedyAnalyticBandLongwave(SG)          # used LW scheme

# Sampling
T_SPINUP    = Day(30)                    # spinup time in days
START_DATE  = DateTime(2000, 1, 1)      # start date of simulation
N_IC        = 2                     # number of initial conditions
SIM_TIME    = 365                   # sampling time in days
SAMPLE_GAP  = 3.65                  # days between sampling

# Perturbation
FAC_PERT_T  = 2f0                   # temperature perturbation amplitude
FAC_PERT_Q  = 0.2f0                 # humidity perturbation amplitude (multiplicative)

# Used standard values
CO2 = 280f0           # default CO2 value 
OCEAN_EM = 1f0        # default ocean emissivity value (SW does not propagate yet)
LAND_EM = 1f0         # default land emissivity value (SW does not propagate yet)



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


# Declare input field container
T_layers        = [Float32[] for _ in 1:NLAYERS]
q_layers        = [Float32[] for _ in 1:NLAYERS]
p               = Float32[]
    #co2        = Float32[]         not needed, because standard value is always used
sst             = Float32[]
lst             = Float32[]
    #lf         = Float32[]         not needed, because land fraction is already in [0,1]
    #ocean_em   = Float32[]         not needed, because standard value is always used
    #land_em    = Float32[]         not needed, because standard value is always used

# Declare output field container
dT_layers       = [Float32[] for _ in 1:NLAYERS]
olw             = Float32[]
slwd            = Float32[] 



### Main loop: Define a simulation, perturb temperature and humidity, run spinup
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
# Create folder
foldername = "zscore_abrlw_$(NAME)"
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
        inputs          = ["Temperature profile, Humidity profile, surface pressure, CO2 concentration, Sea surface temperature, Land surface temperature, Land fraction, Ocean emissivity, Land Emissivity"],
        outputs         = ["Temperature profile tendencies, Outgoing longwave, Surface longwave down"],
        defaults        = ["CO2 concentration, Ocean emissivity, Land Emissivity"],
        defaults_vals   = [CO2, OCEAN_EM, LAND_EM],
    ),

    stats = (;
        input = (;
            order   = ["T", "q", "p", "co2", "sst", "lst", "lf", "ocean_em", "land_em"],
            lengths = [NLAYERS, NLAYERS, 1, 1, 1, 1, 1, 1, 1],

            T_mean          = T_mean,           T_std        = T_std,
            q_mean          = q_mean,           q_std        = q_std,
            p_mean          = p_mean,           p_std        = p_std,
            co2_mean        = co2_mean,         co2_std      = co2_std,
            sst_mean        = sst_mean,         sst_std      = sst_std,
            lst_mean        = lst_mean,         lst_std      = lst_std,
            lf_mean         = lf_mean,          lf_std       = lf_std,
            ocean_em_mean   = ocean_em_mean,    ocean_em_std = ocean_em_std,
            land_em_mean    = land_em_mean,     land_em_std  = land_em_std,
        ),

        output = (;
            order   = ["dT", "olw", "slwd"],
            lengths = [NLAYERS, 1, 1],

            dT_mean         = dT_mean,          dT_std       = dT_std,
            olw_mean        = olw_mean,         olw_std      = olw_std,
            slwd_mean       = slwd_mean,        slwd_std     = slwd_std,
        ),
    ),

    grid = (;
        trunc       = TRUNC,
        nlayers     = NLAYERS,
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
        gap_real    = n_gap * Δt_sec / 3600 / 24,
    ),
    
    perturbation = (;
        fac_pert_t  = FAC_PERT_T,
        fac_pert_q  = FAC_PERT_Q,
    ),
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
    ["Surface Pressure", "Sea Surface Temperature", "Land Surface Temperature"],
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


# Save histograms
histopath = joinpath(folderpath, "histograms")
mkpath(histopath)

CairoMakie.save(joinpath(histopath, "input_T.png"), histo_T)
CairoMakie.save(joinpath(histopath, "input_q.png"), histo_q)
CairoMakie.save(joinpath(histopath, "input_scalar.png"), hist_scalar_in)

CairoMakie.save(joinpath(histopath, "output_dT.png"), hist_dT)
CairoMakie.save(joinpath(histopath, "output_fluxes.png"), hist_fluxes)