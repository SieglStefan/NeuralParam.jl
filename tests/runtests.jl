### NeuralParam test suite
###
### Run directly from the package environment (the dir is `tests/`, not the
### Julia-standard `test/`, so `Pkg.test()` will NOT find it — invoke explicitly):
###
###   julia --project=. tests/runtests.jl
###
### Test tiers (controlled by ENV flags):
###   - fast unit          : pure functions, construction, io, update_ps   (ALWAYS run)
###   - SW integration     : parameterizations inside SpeedyWeather + test_mode training
###                          ON by default; set NEURALPARAM_FULL_TESTS=false to skip
###   - plotting (smoke)    : CairoMakie/GeoMakie figures construct          (opt-in)
###                          NEURALPARAM_TEST_PLOTTING=true
###   - autodiff (Enzyme)   : real reverse-mode AD, ≈1h COLD COMPILE PER SCHEME (opt-in)
###                          per-scheme so you only re-test what you changed:
###                            NEURALPARAM_TEST_AUTODIFF_CONST=true
###                            NEURALPARAM_TEST_AUTODIFF_LINEAR=true      (blocked: data gap)
###                            NEURALPARAM_TEST_AUTODIFF_ABR=true
###                            NEURALPARAM_TEST_AUTODIFF_ABR_GLOBAL=true
###                          or the master switch for the overnight run:
###                            NEURALPARAM_TEST_AUTODIFF=true             (all of the above)

using NeuralParam
using SpeedyWeather
using Lux
using Random
using Dates
using Test

# --- shared fixtures -------------------------------------------------------
const NLAYERS = 8                                   # matches data/stats/zscore_abrlw_L8.jld2
const SG      = SpectralGrid(trunc = 31, nlayers = NLAYERS)

# do the zscore fixtures the neural schemes need actually exist?
const DATA_OK = isfile(joinpath(@__DIR__, "..", "data", "stats", "zscore_abrlw_L$(NLAYERS).jld2"))
DATA_OK || @warn "zscore_abrlw_L$(NLAYERS).jld2 not found — neural-scheme tests will be skipped."

# --- ENV flag helpers ------------------------------------------------------
truthy(v) = lowercase(strip(v)) in ("1", "true", "yes", "on")
# a flag is on if its own var is truthy, OR its group/master var is truthy
envflag(name; group = "") = truthy(get(ENV, name, "")) || (group != "" && truthy(get(ENV, group, "")))

const FULL          = get(ENV, "NEURALPARAM_FULL_TESTS", "true") == "true"
const TEST_PLOTTING = envflag("NEURALPARAM_TEST_PLOTTING")

const AD       = "NEURALPARAM_TEST_AUTODIFF"        # master switch
const AD_CONST  = envflag("$(AD)_CONST";      group = AD)
const AD_LINEAR = envflag("$(AD)_LINEAR";     group = AD)
const AD_ABR    = envflag("$(AD)_ABR";        group = AD)
const AD_ABRG   = envflag("$(AD)_ABR_GLOBAL"; group = AD)
const AD_ANY    = AD_CONST || AD_LINEAR || AD_ABR || AD_ABRG

@testset "NeuralParam.jl" begin

    # ---- fast unit tier (always) ----
    include("test_metrics.jl")
    include("test_stats.jl")
    include("test_utils.jl")
    include("test_models.jl")
    include("test_config.jl")
    include("test_io.jl")
    include("test_parameterizations.jl")

    # ---- SW integration tier ----
    if FULL
        include("test_perturb.jl")
        include("test_parameterizations_sw.jl")
        include("test_training.jl")
    else
        @info "Skipping SW integration tier (NEURALPARAM_FULL_TESTS=false)."
    end

    # ---- plotting smoke tier ----
    if TEST_PLOTTING
        include("test_plotting.jl")
    else
        @info "Skipping plotting smoke tier (set NEURALPARAM_TEST_PLOTTING=true)."
    end

    # ---- autodiff tier (per-scheme, slow) ----
    if AD_ANY
        include("test_autodiff.jl")
    else
        @info "Skipping autodiff tier. Enable per scheme (…_CONST/_LINEAR/_ABR/_ABR_GLOBAL=true) " *
              "or all at once (NEURALPARAM_TEST_AUTODIFF=true)."
    end
end
