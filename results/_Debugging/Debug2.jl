using SpeedyWeather
using NeuralParam
using Random







function perturb_grid_field1!(
    sim,
    var::Symbol; 
    amp,
    offset = 0f0,
    zeromin = false,
    rng = Random.default_rng()
)
    
    SpeedyWeather.initialize!(sim, steps=0)

    field = getfield(sim.variables.grid, var)

    noise = randn!(rng, similar(field))
    field .+= offset .+ amp .* noise

    if zeromin
        field .= max.(field, 0f0)
    end

    SpeedyWeather.set!(sim; var => field)
    SpeedyWeather.initialize!(sim, steps=0)

    return nothing
end


spectral_grid = SpectralGrid()

# temperature only
m = PrimitiveWetModel(; spectral_grid); s = initialize!(m)
perturb_grid_field!(s, :temperature, amp = 2f0)
run!(s, period = Day(1)); @info "temp only"  ex = extrema(s.variables.grid.temperature)

# humidity only
m = PrimitiveWetModel(; spectral_grid); s = initialize!(m)
perturb_grid_field!(s, :humidity, amp = 2f-3, zeromin = true)
run!(s, period = Day(1)); @info "humid only" ex = extrema(s.variables.grid.temperature)