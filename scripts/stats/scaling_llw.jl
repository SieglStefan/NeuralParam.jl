### Script for calculating scaling stats for a NeuralLinearLongwave parameterization
###
### Objective: Load calibrated ConstLinearLW and store:
###     - global parameter a per layer k
###     - global parameter b per layer k
###
### Additionally:
###     - stores and displays a plot of the vertical scaling profile at stats/plots



# Load packages
using Revise
using NeuralParam
using CairoMakie


# Define parameters for storing and loading
NAME = "default"            # name of scaling
SCHEME = "CLLW_default"     # name of the calibrated scheme


# Load ConstLinearLW parameterization
scheme = NeuralParam.load(; path=scheme_dir(SCHEME), file="scheme.jld2")


# Extract parameter and calculate scaling
sc_a = abs.(scheme.ps.a .* scheme.scaling.sc_a) 
sc_b = abs.(scheme.ps.b .* scheme.scaling.sc_b) 



### Store statistics
# Create folder
foldername = "scaling_llw_$(NAME)"
folderpath = prepare_out_dir(stats_dir(), foldername)

# Save reference
NeuralParam.save((sc_a=sc_a, sc_b=sc_b); path=folderpath, file="stats.jld2")
@info "Reference dataset stored at $(folderpath)!"



### Plot loaded scaling parameters
NLAYERS = size(sc_a, 1)
layers = 1:NLAYERS
fig = Figure()

# Parameter a
ax_a = Axis(
    fig[1, 1],
    xlabel = "a",
    ylabel = "Layer",
    title = "Global scaling parameter a",
    yticks = layers,
    yreversed = true,
)

CairoMakie.lines!(ax_a, sc_a, layers)
CairoMakie.scatter!(ax_a, sc_a, layers)

# Parameter b
ax_b = Axis(
    fig[1, 2],
    xlabel = "b",
    ylabel = "Layer",
    title = "Global scaling parameter b",
    yticks = layers,
    yreversed = true,
)

CairoMakie.lines!(ax_b, sc_b, layers)
CairoMakie.scatter!(ax_b, sc_b, layers)

# Display plot
display(fig)

# Store plot
file = "scaling.png"
filepath = joinpath(folderpath, file)
CairoMakie.save(filepath, fig)
