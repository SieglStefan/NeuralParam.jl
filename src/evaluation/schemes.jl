







# Collect saved objects from a results directory into a NamedTuple keyed by name
function collect_objects(names, dir, file)
    return (; (Symbol(n) => first(load(; path = joinpath(dir, n), file)) for n in names)...)
end

collect_schemes(names; dir = joinpath(@__DIR__, "..", "..", "results", "models"),
                       file = "scheme.jld2") = collect_objects(names, dir, file)

collect_rollouts(names; dir = joinpath(@__DIR__, "..", "..", "results", "rollouts"),
                        file = "rollout.jld2") = collect_objects(names, dir, file)