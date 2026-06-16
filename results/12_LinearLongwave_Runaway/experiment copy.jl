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
t_sim = 11
layer = 1
t_spinup = 14





T_llw =     extract_layer(layer, generate_temperature_fields(model_target, model_llw; t_spinup, t_sim))
#T_base =    extract_layer(layer, generate_temperature_fields1(model_target, model_base; t_spinup, t_sim))
T_target =  extract_layer(layer, generate_temperature_fields(model_target, model_target; t_spinup, t_sim))



# Plot comparison of rmse
plot_comparison(T_target, [T_llw], 
                metric=rmse, 
                Δt_sec=Δt_sec, 
                labels=["LLW"],
                title = "RMSE Evolution")

# Plot comparison of bias
plot_comparison(T_target, [T_llw], 
                metric=bias, 
                Δt_sec=Δt_sec, 
                labels=["LLW"],
                title = "Bias Evolution")


plot_comparison(T_target, [T_llw],
                metric=maxdiff,
                Δt_sec=Δt_sec,
                labels=["LLW"],
                title="Max Difference Evolution")


# Plot heatmaps with same colorbar range
plot_heatmaps([T_target[end],  T_llw[end]], titles=["target", "llw"])

plot_heatmap(T_target[end] .-T_llw[end])

