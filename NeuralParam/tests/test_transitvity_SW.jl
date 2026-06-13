using SpeedyWeather
using CairoMakie


# Create spectral grid and models with parameterizations
spectral_grid = SpectralGrid(trunc=31, nlayers=4)


model1 = PrimitiveWetModel(spectral_grid)
model2 = PrimitiveWetModel(spectral_grid)

# Extract timestepping
Δt_sec = model1.time_stepping.Δt_sec



sim1 = initialize!(model1)
sim2 = initialize!(model2)

N = 100

run!(sim1, period = Second(N*Δt_sec))
run!(sim1, period = Second(10*Δt_sec))

run!(sim2, period = Second((N+10)*Δt_sec))


T1 = sim1.variables.grid.temperature[:,4]
T2 = sim2.variables.grid.temperature[:,4]


# Function for plotting a single heatmap of the field F
function plot_heatmapA(  F; 
                        title = "Heatmap", 
                        kwargs...)

    # Create heatmap and plot it                
    fig = CairoMakie.heatmap(
        F;
        title = title,
        kwargs...
    )

    display(fig)

    return fig
end


#plot_heatmaps([T_llw1[end], T_llw2], titles=["generate", "run"])
plot_heatmapA(T1 .- T2)

println("Done")


