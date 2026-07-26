



### Helper functions for loading schemes
# Load model using a name
load_model(name; task = 0) = load_scheme(
    path = joinpath(@__DIR__, "..", "calibration", name, "task_$(task)"),
    file = "scheme.jld2",
)

# Resolve model: If string -> load, if e.g. "nothing", use "nothing"
resolve(spec::AbstractString) = load_model(spec)
resolve(spec) = spec

# Create NamedTuple of schemes needed for evaluation
build_schemes(specs) = (; (k => resolve(v) for (k, v) in specs)...)



### General Parameters
# General
SEED = 42                                   # XXX
EVA_NAME = get(ENV, "EVA_NAME", "")         # from launch.sh / terminal; default = evaluation folder

# Grid
TRUNC = 31                                  #
NLAYERS = 8                                 #

# Model
MODEL = PrimitiveWetModel                   #

# Reference
LW_SCHEME = OneBandLongwave                 #
TRANS = FriersonLongwaveTransmissivity      #

# Perturbation
FAC_PERT_T = 2f0                            #
FAC_PERT_Q = 0.2f0                          #



### General pre-processing
# Set seed
Random.seed!(SEED)

# Spectral grid
SG = SpectralGrid(trunc=TRUNC, nlayers = NLAYERS)

# Saving 
RUN_DIR = joinpath(@__DIR__, EVA_NAME)
mkpath(RUN_DIR)





### Skill Evaluation
SCHEMES_SKILL = build_schemes([             #
    :default => "run_XXX",                  #
    :target => nothing,                     #
    :uncal => nothing,                      #
    :none => nothing,                       #
])
REF_SKILL = load_reference()                #

MAX_HORIZON_SKILL = 31                      #
N_TRAJ_SKILL = 52                            #

HEATMAP_DAYS_SKILL = []
LAYERS_SKILL = []





### Rollout Evaluation
SCHEMES_ROLLOUT = build_schemes([             #
    :default => "run_XXX",                  #
    :target => nothing,                     #
    :uncal => nothing,                      #
    :none => nothing,                       #
])
REF_ROLLOUT = load_reference()                #

MAX_HORIZON_ROLLOUT = 180                      #
N_TRAJ_ROLLOUT = 4                            #

HEATMAP_DAYS_ROLLOUT = [1,7,31,90]
LAYERS_ROLLOUT = [NLAYERS]






### Benchmark Parameters
SCHEMES_BENCHMARK = build_schemes([           #
    :default => "run_XXX",                    #
    :target => nothing,                     #
    :uncal => nothing,                      #
    :none => nothing,                       #
])

N_STEPS = 100





### AB Parameters
MULTI_RUN = "run_XXX"
EXCLUDE = []
N_SCHEMES = 7













