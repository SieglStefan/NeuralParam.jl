### Experiment 1.2: LinearLongwave Runaway
    # Goal: 
    # Problems:
    # Results:
    # Conclusion:


using SpeedyWeather


# Create spectral grid and models with parameterizations
spectral_grid = SpectralGrid(trunc=31, nlayers=1)

radiation_llw = LinearLongwave(a = -5.507875f-8, b = 4.608944f-6)
radiation_llw_0 = LinearLongwave(a=0., b=0.)

model_llw = PrimitiveWetModel(spectral_grid, longwave_radiation = radiation_llw)
model_base = PrimitiveWetModel(spectral_grid, longwave_radiation = radiation_llw_0)
model_target = PrimitiveWetModel(spectral_grid)


# Extract timestepping
Δt_sec = model_llw.time_stepping.Δt_sec


# Run the models and extract the last layer (ground)
t_sim = 14
layer = 1
t_spinup = 14


# Generate a series of temperature fields from a model
function generate_temperature_fields1(   model_base,             # base model used for spinup
                                        model_data;             # data model used generating temperature field
                                        t_spinup = 14,          # spinup time in days for IC sampling
                                        t_sim = 1,              # simulation time in days for which temperature fields are generated
                                        save_every = 1)         # save every n timesteps


    # Copy models to prevent changes in the original models
    model_base_copy = deepcopy(model_base)
    model_data_copy = deepcopy(model_data)


    # Create and spinup the base model simulation
    sim_base = SpeedyWeather.initialize!(model_base_copy)
    run!(sim_base, period = Day(t_spinup))


    # Create data model and simulation
    sim_data = SpeedyWeather.initialize!(model_data_copy)
    copy!(sim_data.variables, sim_base.variables)


    # Create temperature container
    T = Vector{typeof(sim_data.variables.grid.temperature)}()


    # Extract time stepping
    Δt_sec = model_data.time_stepping.Δt_sec

    # Compute the number of timesteps and datapoints for the requested simulation period
    n_steps = Int(round((t_sim*3600*24) / Δt_sec))
    n_data = Int(round(n_steps / save_every))

    # Print information
    @info "Generated $n_data datapoints with temporal distance of $Δt_sec seconds!"


    # Initialize simulation and do a first untracked timestep
    initialize!(sim_data, steps = n_steps+1)
    SpeedyWeather.first_timesteps!(sim_data)

    # Run simulation and save temperature fields
    for step_num in 1:n_steps
        # Timestep
        SpeedyWeather.timestep!(sim_data)

        # Save every "save_every" timestep
        if step_num % save_every == 0
            push!(T, copy(sim_data.variables.grid.temperature))
        end
    end

    return T
end



T_llw =     extract_layer(layer, generate_temperature_fields1(model_target, model_llw; t_spinup, t_sim))
T_base =    extract_layer(layer, generate_temperature_fields1(model_target, model_base; t_spinup, t_sim))
T_target =  extract_layer(layer, generate_temperature_fields1(model_target, model_target; t_spinup, t_sim))



# Plot comparison of rmse
plot_comparison(T_target, [T_base, T_llw], 
                metric=rmse, 
                Δt_sec=Δt_sec, 
                labels=["Base", "LLW"],
                title = "RMSE Evolution")

# Plot comparison of bias
plot_comparison(T_target, [T_base, T_llw], 
                metric=bias, 
                Δt_sec=Δt_sec, 
                labels=["Base", "LLW"],
                title = "Bias Evolution")


# Plot heatmaps with same colorbar range
plot_heatmaps([T_target[end],  T_llw[end], T_base[end]], titles=["target", "llw", "base"])
