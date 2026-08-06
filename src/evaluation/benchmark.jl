### Functions for benchmark evaluation
###
### Evaluates the computational cost of longwave radiation schemes and compares them in a table



# Compares the computational cost of longwave radiation schemes
function evaluate_benchmark(
    scheme;
    spectral_grid,
    model = PrimitiveWetModel,
    n_steps = 10,
)

    # Initialize simulation
    sim = initialize!(model(spectral_grid; longwave_radiation = scheme))
    
    # Initialize steps and do a first timestep
    first_steps!(sim; steps=n_steps+1)


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
    return map(scheme -> evaluate_benchmark(scheme; kwargs...), schemes)
end


# Print benchmark results as a table, one row per scheme
function print_benchmark(results::NamedTuple; baseline = first(keys(results)))

    base = results[baseline].per_step_ms

    println(rpad("scheme", 24), lpad("ms/step", 10), lpad("KB/step", 12),
            lpad("allocs/step", 14), lpad("rel.", 8))
    println("-"^68)

    for (name, r) in pairs(results)
        println(rpad(String(name), 24),
                lpad(round(r.per_step_ms, digits = 3), 10),
                lpad(round(r.per_step_kb, digits = 1), 12),
                lpad(round(Int, r.per_step_allocs), 14),
                lpad(string(round(r.per_step_ms / base, digits = 2), "×"), 8))
    end

    return nothing
end


