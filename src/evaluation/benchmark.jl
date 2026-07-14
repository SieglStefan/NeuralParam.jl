### Functions for benchmark evaluation
###
### XXX



# Compares the computational cost of longwave radiation schemes
function evaluate_benchmark(;
    scheme;
    spectral_grid,
    model_type = PrimitiveWetModel,
    n_steps = 10,
)

    # Initialize simulation
    sim = initialize!(model_type(spectral_grid; longwave_radiation = scheme))
    
    # Initialize steps and do a first timestep
    SpeedyWeather.initialize!(sim, steps = n_steps+1)
    SpeedyWeather.first_timesteps!(sim)


    # Benchmark
    b = @benchmark sim_timesteps!($sim, $n_steps)


    return (;
        per_step_ms     = minimum(b).time   /n_steps /1e6,     # ns -> ms
        per_step_kb     = minimum(b).memory /n_steps /1024,    # bytes -> KB
        per_step_allocs = minimum(b).allocs /n_steps,
    )
end


# Wrapper for a list (NamedTuple) of schemes
function evaluate_benchmark(schemes::NamedTuple; kwargs...)
    return map(scheme -> evaluate_benchmark(;scheme, kwargs...), schemes)
end



