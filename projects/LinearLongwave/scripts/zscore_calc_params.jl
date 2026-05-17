using SpeedyWeather, Statistics

include("training_online.jl")

const TRUNC = 31
const NLAYERS = 4

const SPINUP = 24

const N_IC = 10
const N_STEPS = 100


spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)
model = PrimitiveWetModel(spectral_grid)

# Extract timestepping
(; Δt, Δt_sec) = model.time_stepping
dt = 2Δt

println("Time after spinup (Days): ", Δt_sec*N_STEPS / 3600 / 24)
println("Number of temperature fields: ", N_IC * N_STEPS)


Ts = Float32[]

for i in 1:N_IC
    sim = initialize!(model)

    perturb_temp!(sim)

    run!(sim, period=Hour(SPINUP))

    vars = sim.variables

    for j in 1:N_STEPS

        SpeedyWeather.timestep!(vars, dt, model)

        append!(Ts, vec(vars.grid.temperature))
    end

end


T_mean = mean(Ts)
T_std = std(Ts)


println("T_mean: ", T_mean)
println("T_std: ", T_mean)
