### Debug2_setrun.jl — isolate why set! + run! gives NaN
###
### RUN IN A FRESH JULIA PROCESS (no leftover REPL state):
###   julia --project=C:/Code/sw-ml-thesis-refactor
###   julia> include("results/_Debugging/Debug2_setrun.jl")
###
### Pure SpeedyWeather only. NeuralParam is intentionally NOT loaded,
### so if this NaNs, the issue is SpeedyWeather usage, not your package.

using SpeedyWeather
using Dates

const SG = SpectralGrid(trunc = 31, nlayers = 8)

# fresh, independent simulation for every test
freshsim() = initialize!(PrimitiveWetModel(; spectral_grid = SG))

# run a labelled test, never let one failure stop the file
function check(label, build)
    try
        sim = build()
        T   = sim.variables.grid.temperature
        @info label  finite = all(isfinite, T)  extrema = extrema(T)
    catch e
        @warn "$label  ERRORED"  exception = (e, catch_backtrace())
    end
end

@info "=== Debug2: set! + run! isolation ==="

# 1. CONTROL: plain run, no set! at all. MUST be finite — if not, the model
#    setup itself is the problem, not set!.
check("1  plain run! Day(1)", () -> begin
    sim = freshsim()
    run!(sim, period = Day(1))
    sim
end)

# 2. set! temperature to ITS OWN value, DEFAULT lf, then run.
check("2  set!(temperature=T) default-lf + run", () -> begin
    sim = freshsim()
    T = copy(sim.variables.grid.temperature)
    set!(sim; temperature = T)
    run!(sim, period = Day(1))
    sim
end)

# 3. same, but lf = 2 (what SpeedyWeather's own tests use)
check("3  set!(temperature=T, lf=2) + run", () -> begin
    sim = freshsim()
    T = copy(sim.variables.grid.temperature)
    set!(sim; temperature = T, lf = 2)
    run!(sim, period = Day(1))
    sim
end)

# 4. same, but lf = 1 explicit
check("4  set!(temperature=T, lf=1) + run", () -> begin
    sim = freshsim()
    T = copy(sim.variables.grid.temperature)
    set!(sim; temperature = T, lf = 1)
    run!(sim, period = Day(1))
    sim
end)

# 5. set BOTH leapfrog levels before running
check("5  set! temperature lf=1 AND lf=2 + run", () -> begin
    sim = freshsim()
    T = copy(sim.variables.grid.temperature)
    set!(sim; temperature = T, lf = 1)
    set!(sim; temperature = T, lf = 2)
    run!(sim, period = Day(1))
    sim
end)

# 6. is it temperature-specific? set! HUMIDITY to itself, default lf
check("6  set!(humidity=q) default-lf + run", () -> begin
    sim = freshsim()
    q = copy(sim.variables.grid.humidity)
    set!(sim; humidity = q)
    run!(sim, period = Day(1))
    sim
end)

# 7. does set! ALONE (no run) keep things finite?
check("7  set!(temperature=T) then NO run, check grid", () -> begin
    sim = freshsim()
    T = copy(sim.variables.grid.temperature)
    set!(sim; temperature = T)
    sim
end)

# 8. shorter run after set! — does it blow up immediately or over time?
check("8  set!(temperature=T) + run Hour(1)", () -> begin
    sim = freshsim()
    T = copy(sim.variables.grid.temperature)
    set!(sim; temperature = T)
    run!(sim, period = Hour(1))
    sim
end)

@info "=== Debug2 done ==="
