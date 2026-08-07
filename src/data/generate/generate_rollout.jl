### Generation of rollouts for evaluation
###
### Objective: Initialize a scheme from reference states at many start days, propagate it and
###            compare it to the reference at every lead time
###
### Stored per rollout:
###  - scores: metrics per lead day, trajectory and column entry
###             -example:    scores.T.rmse[h, i, k]
###                                        │  │  └── layer
###                                        │  └───── which forecast (1…52)
###                                        └──────── how many days into it (1…31)
###             -metrics:    rmse, bias, maxdiff   rollout vs reference (rmse/bias area-weighted)
###                          mean                  global mean of the ROLLOUT itself
###  - heatmap_states / heatmap_ref: full fields of one trajectory at selected lead days



# Default probes: the fields a longwave scheme is judged on
const DEF_PROBES = (;
    T    = v -> SpeedyWeather.get_step(v.grid.temperature),
    olw  = v -> v.parameterizations.outgoing_longwave,
    slwd = v -> v.parameterizations.surface_longwave_down,
    slwu = v -> v.parameterizations.surface_longwave_up,
)
# Units of the default probes, for axis labels
const DEF_PROBE_UNITS = (; T = "K", olw = "W/m²", slwd = "W/m²", slwu = "W/m²")


# Function for sampling probes along a trajectory of sim, starting with initial_condition
function sample_trajectory(
    sim,
    probes,
    initial_condition;
    n_samples, n_gap
)

    # Initialize sim, do a first step and set initial condition
    first_steps!(sim; steps = n_samples * n_gap + 1)
    copy!(sim.variables, initial_condition)

    # Create container for probed fields with a first entry
    data = (; (p => [copy(probe(sim.variables))] for (p, probe) in pairs(probes))...)


    # Loop over samples
    for _ in 1:n_samples

        # Do n_gap steps
        sim_timesteps!(sim, n_gap)

        # Store probed fields
        for (p, probe) in pairs(probes)
            push!(data[p], copy(probe(sim.variables)))
        end
    end

    return data
end



# Utility functions for the number of column entries (nlayers for profiles, 1 for scalars)
n_cols(f::AbstractVector) = 1
n_cols(f::AbstractMatrix) = size(f, 2)

# Utility functions for extracting a column from a field
get_col(f::AbstractVector, k) = f
get_col(f::AbstractMatrix, k) = f[:, k]



# Generate a rollout of one scheme against a reference and store it as rollout.jld2
function generate_rollout(;
    spectral_grid,  # used spectral grid
    name,           # name of the rollout
    dir,            # output directory

    scheme,         # scheme to roll out (object, or name of a stored scheme)
    reference,      # reference data set to compare against
    model,          # model used
    probes,         # fields the rollout is judged on

    max_horizon,    # maximum forecast length in days
    n_traj,         # number of trajectories sampled
    rollout_t,      # period the trajectory start days are spread over

    heatmap_days,   # lead days for which entire fields are returned
    heatmap_traj,   # trajectory the returned fields are taken from
)

    # Load the scheme if it is given by name
    lw_scheme = resolve_scheme(scheme)

    # Grid area weights, so the metrics match the ones logged during training
    w = area_weights(spectral_grid)

    # Extract time step and calculate necessary steps for one day
    steps_per_day = steps_from_days(1, model(spectral_grid).time_stepping.Δt)

    # Calculate the start day of every trajectory, e.g. (0, 7, 14, ..., 350, 357)
    start_days = round.(Int, range(0, rollout_t, length = n_traj + 1))[1:end-1]


    ### Roll out every trajectory and reduce it against the reference
    rollout = with_reference(reference) do ref

        # Check the reference covers the whole rollout
        n_needed = maximum(start_days) + max_horizon
        ref.sim_days ≥ n_needed || error(
            "Reference '$reference' covers $(ref.sim_days) days, need $n_needed " *
            "(rollout_t = $rollout_t, max_horizon = $max_horizon, n_traj = $n_traj)."
        )

        # Learn the shape of every probed field from the reference (nlayers or 1)
        ncols = map(probe -> n_cols(probe(ref[0])), probes)

        # Prepare containers for the metrics
        curve(nc) = zeros(Float64, max_horizon, n_traj, nc)
        scores = map(nc -> (; rmse    = curve(nc), bias = curve(nc),
                           maxdiff = curve(nc), mean = curve(nc)), ncols)

        # Prepare containers for the heatmap fields of one trajectory
        heatmap_states = nothing
        heatmap_ref    = nothing


        # Loop over all start days
        for (i, s) in enumerate(start_days)

            # Initialize simulation
            sim = initialize!(model(spectral_grid; longwave_radiation = lw_scheme))

            # Propagate the scheme, starting from the reference state at day s
            traj = sample_trajectory(sim, probes, ref[s];
                                     n_samples = max_horizon, n_gap = steps_per_day)

            # Loop over lead times
            for h in 1:max_horizon

                # Reference state at day s+h
                ref_state = ref[s + h]

                # Loop over probed fields
                for (p, probe) in pairs(probes)

                    f_run = traj[p][h+1]            # rollout at lead day h
                    f_ref = probe(ref_state)        # reference at day s+h

                    # Loop over column entries (layers for profiles, one pass for scalars)
                    for k in 1:ncols[p]
                        x, y = get_col(f_run, k), get_col(f_ref, k)

                        # Error of the rollout against the reference
                        scores[p].rmse[h,i,k]    = wrmse(x, y, w)
                        scores[p].bias[h,i,k]    = wbias(x, y, w)
                        scores[p].maxdiff[h,i,k] = maxdiff(x, y)

                        # Global mean of the rollout (drift diagnostic)
                        scores[p].mean[h,i,k]    = wmean(x, w)
                    end
                end
            end

            # Keep the full fields of one trajectory for the heatmaps
            if i == heatmap_traj
                heatmap_states = map(fields -> [fields[d+1] for d in heatmap_days], traj)
                heatmap_ref    = map(probe -> [probe(ref[s + d]) for d in heatmap_days], probes)
            end

            @info "Trajectory $i/$(n_traj) (start day $s) finished!"
        end


        # Collect rollout results
        return (;
            days            = 1:max_horizon,
            probes          = keys(probes),
            start_days      = start_days,

            scores          = scores,

            heatmap_days    = heatmap_days,
            heatmap_traj    = heatmap_traj,
            heatmap_states  = heatmap_states,
            heatmap_ref     = heatmap_ref,
        )
    end


    ### Save reduced rollout
    save(rollout; dir, file = "rollout.jld2")
    @info "Rollout dataset $(name) stored at $(dir)!"

    return rollout
end
