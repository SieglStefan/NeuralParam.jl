### Script for calculating scaling stats for a NeuralLinearLongwave parameterization
###
### Objective: Load calibrated ConstLinearLW and store:
###     - global parameter a per layer k
###     - global parameter b per layer k
###
### Additionally:
###     - stores and displays a plot of the vertical scaling profile at stats/plots



### Load packages
using Revise
using NeuralParam
using JLD2
using CairoMakie




# Choose calibrated model and define number of vertical layers
NAME = ""             # name of the scaling stats file
scheme_run = "run_T31_L8_2026-07-07_23-22-44"
PARAM_NAME = "scheme.jld2"       # name of the parameterization file
NLAYERS = 8           # number of vertical layers


# Folder for stats files (.jld2, .png)
foldername = "scaling_llw_L$(NLAYERS)$(NAME)"
folderpath = joinpath(@__DIR__, "..", "..", "data", "stats", foldername)
mkpath(folderpath)



### Load ConstLinearLW parameterization
# Create parameterization filepath
path = joinpath(@__DIR__, "..", "..", "results", "1_ConstLinearLW", "calibration", scheme_run)

# Load parameterization
scheme = NeuralParam.load_scheme(; path, file=PARAM_NAME)

# Extract parameter and calculate scaling
sc_a = abs.(scheme.ps.a .* scheme.scaling.sc_a) 
sc_b = abs.(scheme.ps.b .* scheme.scaling.sc_b) 




### Store statistics
file = "stats.jld2"
filepath = joinpath(folderpath, file)

JLD2.jldsave(filepath; sc_a, sc_b)



### Plot loaded scaling parameters
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
