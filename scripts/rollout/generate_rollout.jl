### Script for generating rollouts for evaluation
###
### Objective: Propagate a specific scheme for certain horizons over a year and reduce it
###            against a reference data set



### Load packages
using Revise
using NeuralParam
using SpeedyWeather
using Dates
using Random



### Define spectral grid
TRUNC = 31
NLAYERS = 8
SG = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)



### Define common parameters for task selection
# Surface emissivity
EM_OCEAN    = 0.98f0
EM_LAND     = 0.98f0

# Schemes
OBLW        = OneBandLongwave(SG;
                    transmissivity     = FriersonLongwaveTransmissivity(SG),
                    radiative_transfer = OneBandLongwaveRadiativeTransfer(SG;
                                            emissivity_ocean = EM_OCEAN,
                                            emissivity_land  = EM_LAND),)
# References
REF_OBLW    = "OBLW_default"

# Trajectory protocols
SKILL       = (; max_horizon = 31,  n_traj = 52, heatmap_days = [0,1,3,7,14])
STAB        = (; max_horizon = 180, n_traj = 4,  heatmap_days = [1,30,90,180])



### Define possible variants for rollouts
variants = [

    ### 0. Test variant
    (;  name            = "TEST_ROLLOUT",
        scheme          = ZeroLW(),
        reference       = REF_OBLW,
        max_horizon     = 10,
        n_traj          = 2,
        rollout_t       = 20,
        heatmap_days    = [1,10],
        heatmap_traj    = 1,
    ),



    ### ZeroLW (no longwave parameterization)
    # 1. Skill
    (;  name            = "ZeroLW_default_skill",
        scheme          = ZeroLW(),
        reference       = REF_OBLW,
        SKILL...
    ),
    # 2. Stability
    (;  name            = "ZeroLW_default_stab",
        scheme          = ZeroLW(),
        reference       = REF_OBLW,
        STAB...
    ),



    ### Default OneBandLongwave (for sanity test, should lead to zero error in evaluation)
    # 3. Skill
    (;  name            = "OBLW_default_skill",
        scheme          = OBLW,
        reference       = REF_OBLW,
        SKILL...
    ),
    # 4. Stability
    (;  name            = "OBLW_default_stab",
        scheme          = OBLW,
        reference       = REF_OBLW,
        STAB...
    ),



    ### Default ConstLW rollouts
    ## Direct output
    # 5. Skill
    (;  name            = "CLLW_direct_default_skill",
        scheme          = "CLLW_direct_default",
        reference       = REF_OBLW,
        SKILL...
    ),
    # 6. Stability
    (;  name            = "CLLW_direct_default_stab",
        scheme          = "CLLW_direct_default",
        reference       = REF_OBLW,
        STAB...
    ),

    ## Linear output
    # 7. Skill
    (;  name            = "CLLW_linear_default_skill",
        scheme          = "CLLW_linear_default",
        reference       = REF_OBLW,
        SKILL...
    ),
    # 8. Stability
    (;  name            = "CLLW_linear_default_stab",
        scheme          = "CLLW_linear_default",
        reference       = REF_OBLW,
        STAB...
    ),

    ## Planck output
    # 9. Skill
    (;  name            = "CLLW_planck_default_skill",
        scheme          = "CLLW_planck_default",
        reference       = REF_OBLW,
        SKILL...
    ),
    # 10. Stability
    (;  name            = "CLLW_planck_default_stab",
        scheme          = "CLLW_planck_default",
        reference       = REF_OBLW,
        STAB...
    ),



    ### Default NeuralLW rollouts
    ## Direct output
    # 11. Skill
    (;  name            = "NLLW_direct_default_skill",
        scheme          = "NLLW_direct_default",
        reference       = REF_OBLW,
        SKILL...
    ),
    # 12. Stability
    (;  name            = "NLLW_direct_default_stab",
        scheme          = "NLLW_direct_default",
        reference       = REF_OBLW,
        STAB...
    ),

    ## Linear output
    # 13. Skill
    (;  name            = "NLLW_linear_default_skill",
        scheme          = "NLLW_linear_default",
        reference       = REF_OBLW,
        SKILL...
    ),
    # 14. Stability
    (;  name            = "NLLW_linear_default_stab",
        scheme          = "NLLW_linear_default",
        reference       = REF_OBLW,
        STAB...
    ),

    ## Planck output
    # 15. Skill
    (;  name            = "NLLW_planck_default_skill",
        scheme          = "NLLW_planck_default",
        reference       = REF_OBLW,
        SKILL...
    ),
    # 16. Stability
    (;  name            = "NLLW_planck_default_stab",
        scheme          = "NLLW_planck_default",
        reference       = REF_OBLW,
        STAB...
    ),
]

# Get task number and choose task
task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
v = variants[task+1]



### Define and extract parameters
# General
NAME         = get(v, :name, "NoName_default")          # name of the rollout

# Scheme and reference
SCHEME       = get(v, :scheme, ZeroLW())                # scheme to roll out
REFERENCE    = get(v, :reference, REF_OBLW)             # reference for comparison

# Model
MODEL        = get(v, :model, PrimitiveWetModel)        # used model
PROBES       = get(v, :probes, DEF_PROBES)              # probed fields

# Rollout
MAX_HORIZON  = get(v, :max_horizon, 31)                 # maximum forecast length in days
N_TRAJ       = get(v, :n_traj, 52)                      # number of trajectories sampled
ROLLOUT_T    = get(v, :rollout_t, 365)                  # sampling time of rollout

# Heatmaps
HEATMAP_DAYS = get(v, :heatmap_days, [1,7,31])          # lead days for which entire fields are returned
HEATMAP_TRAJ = get(v, :heatmap_traj, 1)                 # choice of specific trajectory for heatmap



### Prepare generation
# Create output folder
out_dir = startswith(NAME, "TEST") ? fresh_out_dir : prepare_out_dir
DIR = out_dir(rollout_dir(), NAME)



### Generate the rollout
rollout = generate_rollout(;
    spectral_grid   = SG,
    name            = NAME,
    dir             = DIR,

    scheme          = SCHEME,
    reference       = REFERENCE,
    model           = MODEL,
    probes          = PROBES,

    max_horizon     = MAX_HORIZON,
    n_traj          = N_TRAJ,
    rollout_t       = ROLLOUT_T,

    heatmap_days    = HEATMAP_DAYS,
    heatmap_traj    = HEATMAP_TRAJ,
)



### Create and store info.toml file
write_info(;
    dir = DIR,
    file = "info.toml",

    general = (;
        name        = NAME,
        created     = now(),
        julia       = string(VERSION),
        sw_vers     = string(pkgversion(SpeedyWeather)),
    ),

    grid = (;
        trunc       = TRUNC,
        nlayers     = NLAYERS,
        grid_type   = string(nameof(SG.Grid)),
    ),

    rollout = (;
        model        = string(nameof(MODEL)),
        scheme       = scheme_name(SCHEME),
        reference    = REFERENCE,
        probes       = [string(p) for p in keys(PROBES)],
        metrics      = ["rmse", "bias", "maxdiff", "mean"],
    ),

    sampling = (;
        max_horizon  = MAX_HORIZON,
        n_traj       = N_TRAJ,
        rollout_t    = ROLLOUT_T,
        start_days   = rollout.start_days,
        heatmap_days = HEATMAP_DAYS,
        heatmap_traj = HEATMAP_TRAJ,
    ),
)
