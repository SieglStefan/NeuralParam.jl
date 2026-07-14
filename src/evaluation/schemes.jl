### XXX


# Load model from a calibration run folder
load_model(run_path, task = 0) = load_scheme(
    path = joinpath(run_path, "task_$(task)"), 
    file = "scheme.jld2"
)

# Resolve spec: String -> load, otherwise pass on  (e.g. "nothing")
#   String:             - path to run folder, loads task 0
#   (path, task)        - loads specific task
#   scheme/nothing      - pass it on unchanged
resolve(spec::AbstractString) = load_model(spec)
resolve(spec::Tuple)          = load_model(spec...)
resolve(spec)                 = spec

# Build schemes
build_schemes(specs, dir) = (; (k => resolve(v, dir) for (k, v) in specs)...)