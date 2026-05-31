### Script for calculating mean and std data for zscore-transformation for 
### NN inputs for a AnalyticBandRadiation.jl emulator parameterization
###
### Objective: Calculate global mean and std for:
### - temperature for every layer k
### - humidity for every layer k
### - surface pressure
### - sea surface temperature
### - land surface temperature
### - land fraction



using SpeedyWeather, AnalyticBandRadiation
const SpeedyExt = Base.get_extension(AnalyticBandRadiation,
                                     :AnalyticBandRadiationSpeedyWeatherExt)
using Statistics, Random, Dates
using JLD2
using CairoMakie

include(joinpath(@__DIR__, "..", "..", "src", "utils", "data.jl"))

Random.seed!(1234)

const TRUNC = 31
const NLAYERS = 8

const T_SPINUP = 14     # days
const N_IC = 5          # number of initial conditions
const N_STEPS = 20      # samples per IC
const N_GAP = 5         # timesteps between samples
const AMP_T = 2f0       # temperature perturbation amplitude (additive)
const AMP_Q = 0.05f0    # humidity perturbation amplitude (multiplicative)



# Define spectral grid and model
spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
longwave = SpeedyExt.SpeedyAnalyticBandLongwave(spectral_grid)
model = PrimitiveWetModel(spectral_grid; longwave_radiation = longwave)

# Constant AnalyticBandRadiation parameters
geometry  = SpeedyExt._speedy_column_geometry(model)
constants = SpeedyExt._speedy_physical_constants(model)

# Create template simulation
sim_temp = initialize!(model)

# Extract timestepping
(; Δt_sec) = model.time_stepping

# Print information
println("--------------------------------------------------------------------")
println("Total time propagated per IC (days): ", T_SPINUP+((Δt_sec*(N_STEPS*N_GAP+1)) / 3600) / 24)
println("Time between samples (hours): ", N_GAP * Δt_sec / 3600)
println("Total number of global temperature fields: ", N_IC * N_STEPS)
println("--------------------------------------------------------------------")



# Declare field container
T_layers = [Float32[] for _ in 1:NLAYERS]
q_layers = [Float32[] for _ in 1:NLAYERS]
ps = Float32[]
sst = Float32[]
lst = Float32[]
lf = Float32[] 

dT_layers = [Float32[] for _ in 1:NLAYERS] 



# Main loop: Define a simulation, perturb temperature, run spinup
for i in 1:N_IC

    # Create simulation
    sim = deepcopy(sim_temp)
    
    # Perturbate temperature and humidity field
    perturb_grid_temp!(sim; amp=AMP_T)
    perturb_grid_humid!(sim; amp=AMP_Q)

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
        
        vars = sim.variables



        # Collect input fields
        for k in 1:NLAYERS
            append!(T_layers[k], vec(vars.grid.temperature[:, k]))
            append!(q_layers[k], vec(vars.grid.humidity[:, k]))
        end

        append!(ps,  vec(vars.grid.pressure))
        append!(sst, vec(vars.prognostic.ocean.sea_surface_temperature))
        append!(lst, vec(vars.prognostic.land.soil_temperature[:, 1]))
        append!(lf,  vec(model.land_sea_mask.mask))



        # Collect output fields by explicitely calling the parameterization for tendencies
        for ij in axes(vars.grid.temperature_prev, 1)

            T_col = @view vars.grid.temperature_prev[ij, :]
            q_col = @view vars.grid.humidity_prev[ij, :]
            Φ_col = @view vars.grid.geopotential[ij, :]
            p_col = vars.grid.pressure_prev[ij]

            surface = SurfaceState{Float32}(
                sea_surface_temperature  = vars.prognostic.ocean.sea_surface_temperature[ij],
                land_surface_temperature = vars.prognostic.land.soil_temperature[ij, 1],
                land_fraction            = model.land_sea_mask.mask[ij],
            )

            profile = AtmosphereProfile(
                temperature      = T_col,
                humidity         = q_col,
                geopotential     = Φ_col,
                surface_pressure = p_col,
                CO₂              = longwave.default_CO₂,
            )

            dTdt = zeros(Float32, NLAYERS)
            diag = LongwaveDiagnostics{Float32}()

            solve_longwave!(
                dTdt,
                diag,
                longwave.scheme,
                profile,
                geometry,
                surface,
                constants,
            )

            for k in 1:NLAYERS
                push!(dT_layers[k], Float32(dTdt[k]))
            end
        end
    end

    println("\tIC $i finished")
end



# Calculate mean and std of fields
T_mean = Float32[mean(T_layers[k]) for k in 1:NLAYERS]
T_std  = Float32[std(T_layers[k])  for k in 1:NLAYERS]

q_mean = Float32[mean(q_layers[k]) for k in 1:NLAYERS]
q_std  = Float32[std(q_layers[k])  for k in 1:NLAYERS]

ps_mean  = Float32[mean(ps)]
ps_std   = Float32[std(ps)]

sst_mean = Float32[mean(sst)]
sst_std  = Float32[std(sst)]

lst_mean = Float32[mean(lst)]
lst_std  = Float32[std(lst)]

lf_mean  = Float32[mean(lf)]
lf_std   = Float32[std(lf)]

dT_mean = Float32[mean(dT_layers[k]) for k in 1:NLAYERS]
dT_std  = Float32[std(dT_layers[k])  for k in 1:NLAYERS]

# Combine to input mean and std
input_mean = vcat(T_mean, q_mean, ps_mean, sst_mean, lst_mean, lf_mean)
input_std  = vcat(T_std,  q_std,  ps_std,  sst_std,  lst_std,  lf_std)

# Combine to output mean and std
output_mean = dT_mean
output_std  = dT_std



# Store statistics
file = "abrlw_T$(TRUNC)_L$(NLAYERS).jld2"
path = joinpath(@__DIR__, "..", "..", "data", "zscore", file)

mkpath(dirname(path))

JLD2.jldsave(path;
    input_mean,
    input_std,
    output_mean,
    output_std
)



# Plot vector results
layers = 1:NLAYERS

fig = Figure()

# Temperature plot
ax_T = Axis(
    fig[1, 1],
    xlabel = "Temperature (K)",
    ylabel = "Layer",
    title = "Global mean temperature profile",
    yticks = layers,
    yreversed = true,
)

lines!(ax_T, T_mean, layers)
scatter!(ax_T, T_mean, layers)
errorbars!(ax_T, T_mean, layers, T_std; direction = :x, whiskerwidth = 8)

# Humidity plot
ax_q = Axis(
    fig[1, 2],
    xlabel = "Specific humidity (kg/kg)",
    ylabel = "Layer",
    title = "Global mean humidity profile",
    yticks = layers,
    yreversed = true,
)

lines!(ax_q, q_mean, layers)
scatter!(ax_q, q_mean, layers)
errorbars!(ax_q, q_mean, layers, q_std; direction = :x, whiskerwidth = 8)

display(fig)



# Print single results:
println("Results:")
println("")
println("\t\t surface pressure mean: $ps_mean")
println("\t\t surface pressure std: $ps_std")
println("")
println("\t\t sea surface temperature mean: $sst_mean")
println("\t\t sea surface temperature std: $sst_std")
println("")
println("\t\t land surface temperature mean: $lst_mean")
println("\t\t land surface temperature std: $lst_std")
println("")
println("\t\t land fraction mean: $lf_mean")
println("\t\t land fraction std: $lf_std")
println("")