




function perturb_grid_field!(
    sim,
    var::Symbol; 
    fac_add = 0f0,
    fac_mult = 0f0,
    offset = 0f0,
    zeromin = false,
    rng = Random.default_rng()
)
    
    SpeedyWeather.initialize!(sim, steps=0)

    field = copy(getfield(sim.variables.grid, var))


    field .+= fac_add .* randn!(rng, similar(field))

    field .*= 1f0 .+ fac_mult .* randn!(rng, similar(field))

    field .+= offset

    if zeromin
        field .= max.(field, 0f0)
    end

    SpeedyWeather.set!(sim; var => field)
    SpeedyWeather.initialize!(sim, steps=0)

    return nothing
end