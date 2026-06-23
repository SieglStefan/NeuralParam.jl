### Perturbation helper inside SpeedyWeather.
### Guards the bug we chased for a long time: perturb + spin-up must stay finite
### (grid populated before reading, no NaN blow-up during integration).

@testset "perturbation" begin
    model = PrimitiveWetModel(; spectral_grid = SG)
    sim   = initialize!(model)

    # temperature additive, humidity multiplicative (relative)
    perturb_grid_field!(sim, :temperature; fac_add  = 2f0)
    perturb_grid_field!(sim, :humidity;    fac_mult = 0.2f0, zeromin = true)

    @test all(isfinite, Array(parent(sim.variables.grid.temperature)))
    @test all(isfinite, Array(parent(sim.variables.grid.humidity)))

    # must survive a short spin-up without blowing up
    run!(sim, steps = 2)
    @test all(isfinite, Array(parent(sim.variables.grid.temperature)))
end
