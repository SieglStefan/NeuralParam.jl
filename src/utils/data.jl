### Data utilities
###
### Helper functions for handling data



# Function for perturbing a grid variable field of a simulation
function perturb_grid_field!(
    sim,
    var::Symbol; 
    fac_add = 0f0,
    fac_mult = 0f0,
    offset = 0f0,
    zeromin = false,
    rng = Random.default_rng()
)
    
    # Initalize simulation (fill variables.grid if not initialized yet)
    SpeedyWeather.initialize!(sim, steps=0)

    # Copy field for perturbation
    field = copy(getfield(sim.variables.grid, var))


    # Additive perturbation
    field .+= fac_add .* randn!(rng, similar(field))

    # Multiplicative perturbation
    field .*= 1f0 .+ fac_mult .* randn!(rng, similar(field))

    # Offset
    field .+= offset

    # Only take positive values if set
    if zeromin
        field .= max.(field, 0f0)
    end


    # Set variables onto the simulation and initialize again to apply
    SpeedyWeather.set!(sim; var => field)
    SpeedyWeather.initialize!(sim, steps=0)

    return nothing
end