using SpeedyWeather, CairoMakie, JLD2

const TRUNC = 31
const N_LAYERS = 8

const SPINUP = Hour(1)
const N_IC = 1
const N_DATA = 1
const N_GAP = 1

spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=N_LAYERS)
model = PrimitiveWetModel(spectral_grid)
sim0 = initialize!(model)


dt_sec = model.time_stepping.Δt_sec
println("\nTime of Δt in hours: \t\t\t", dt_sec/3600)
println("Duration of gap in hours: \t\t", N_GAP * dt_sec/3600)
println("Span of time of sampling in days: \t", N_DATA * N_GAP * dt_sec / (24*3600), "\n")

n_gridpoints = length(initialize!(model).variables.grid.temperature[:, 1])
n_samples = N_IC * N_DATA * n_gridpoints

X = Array{Float32}(undef, n_samples, N_LAYERS)
Y = Array{Float32}(undef, n_samples, N_LAYERS)


for n_ic in 1:N_IC

    simulation = initialize!(model)

    # Perturbate initial conditions
    vars = simulation.variables
    vars.prognostic.vorticity .+= 1f-8 .* randn(ComplexF32, size(vars.prognostic.vorticity))

    # Spinup model
    run!(simulation, period=SPINUP)

    # Begin data sampling

    for i in 1:N_DATA

        for j in 1:N_GAP
            run!(simulation, period=Second(dt_sec))
        end

        # Sample data
        T = Array(simulation.variables.grid.temperature)
        Q = Array(simulation.variables.tendencies.grid.temperature)

        @assert all(isfinite, T)
        @assert all(isfinite, Q)

        # Store data
        # XXX

    end
end

using JLD2



jldsave("linear_longwave_dataset.jld2";
    X,
    Y,
    metadata = Dict(
        "trunc" => TRUNC,
        "n_layers" => N_LAYERS,
        "spinup" => string(SPINUP),
        "n_ic" => N_IC,
        "n_data" => N_DATA,
        "n_gap" => N_GAP,
    )
)