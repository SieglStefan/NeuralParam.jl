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
using Dates


# Define parameters for storing and loading
NAME = "default"            # name of scaling
SCHEME = "CLLW_default"     # name of the calibrated scheme


# Load ConstLinearLW parameterization
scheme = NeuralParam.load(; path=scheme_dir(SCHEME), file="scheme.jld2")


# Extract parameter and calculate scaling and number of vertical layers
sc_a = abs.(scheme.ps.a .* scheme.scaling.sc_a) 
sc_b = abs.(scheme.ps.b .* scheme.scaling.sc_b) 
nlayers = size(sc_a, 1)



### Store statistics
# Create folder
foldername = "scaling_llw_$(NAME)"
folderpath = fresh_out_dir(stats_dir(), foldername)

# Save stats
NeuralParam.save((sc_a=sc_a, sc_b=sc_b); path=folderpath, file="stats.jld2")
@info "Statistics $(foldername) stored at $(folderpath)!"


# Create and store info.toml file
write_info(;
    path = folderpath,
    file = "info.toml",

    general = (;
        created = now(),
        scheme  = SCHEME,
    ),

    stats = (;
        input = (;
            order   = [],
            lengths = [],
        ),

        output = (;
            order   = ["sc_a", "sc_b"],
            lengths = [nlayers, nlayers],
            sc_a = sc_a, sc_b = sc_b,
        ),
    ),
)



### Plot loaded scaling parameters
layers = 1:nlayers
fig = Figure()

# Parameter a
ax_a = Axis(
    fig[1, 1],
    xlabel = "a",
    ylabel = "Layer",
    title = "Global scaling parameter a",
    yticks = layers,
    xticks = LinearTicks(3),
    yreversed = true,
)

CairoMakie.lines!(ax_a, sc_a, layers)
CairoMakie.scatter!(ax_a, sc_a, layers)
CairoMakie.vlines!(ax_a, [0]; color = :gray, linestyle = :dash)
CairoMakie.vlines!(ax_a, [5f-8]; color = :red, linestyle = :dash)

# Parameter b
ax_b = Axis(
    fig[1, 2],
    xlabel = "b",
    ylabel = "Layer",
    title = "Global scaling parameter b",
    yticks = layers,
    xticks = LinearTicks(3),
    yreversed = true,
)

CairoMakie.lines!(ax_b, sc_b, layers)
CairoMakie.scatter!(ax_b, sc_b, layers)
CairoMakie.vlines!(ax_b, [0]; color = :gray, linestyle = :dash)
CairoMakie.vlines!(ax_b, [5f-6]; color = :red, linestyle = :dash)

# Display plot
display(fig)

# Store plot
file = "scaling.png"
filepath = joinpath(folderpath, file)
CairoMakie.save(filepath, fig)
