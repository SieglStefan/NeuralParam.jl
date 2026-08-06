### Generation of rollouts for evaluation
###
### Objective: Initialize a scheme from reference states at many start days, propagate it and
###            compare it to the reference at every lead time
###
### Stored per rollout:
###     - err:  area-weighted rmse/bias, one entry per lead day, trajectory and column entry
###     - glob: area-weighted global means of rollout and reference, for drift diagnostics
###     - heatmap_states / heatmap_ref: full fields of one trajectory at selected lead days



# Number of column entries of a probed field (layers for profiles, 1 for scalars)
n_cols(f::AbstractVector) = 1
n_cols(f::AbstractMatrix) = size(f, 2)

# One column entry of a probed field
#   - copied instead of viewed, so the result broadcasts against the plain area weight vector
get_col(f::AbstractVector, k) = f
get_col(f::AbstractMatrix, k) = f[:, k]



# Generate a rollout of one scheme against a reference and store it as rollout.jld2
function generate_rollout(
    spectral_grid;
    name         = "default",               # name of the rollout
    dir          = rollout_dir(name),       # output directory

    scheme       = nothing,                 # scheme to roll out (object, or name of a stored scheme)
    reference    = "OBLW_default",          # reference data set to compare against
    model        = PrimitiveWetModel,       # model used
    probes       = PROBES,                  # fields the rollout is judged on

    max_horizon  = 31,                      # maximum forecast length in days
    n_traj       = 52,                      # number of trajectories sampled
    rollout_t    = 365,                     # period the trajectory start days are spread over

    heatmap_days = [1, 7, 31],              # lead days for which entire fields are returned
    heatmap_traj = 1,                       # trajectory the returned fields are taken from
)

    # Load the scheme if it is given by name
    lw_scheme = resolve_scheme(scheme)

    # Grid area weights, so the metrics match the ones logged during training
    w = area_weights(spectral_grid)

    # Extract time step and calculate necessary steps for one day
    (; Δt) = initialize!(model(spectral_grid; longwave_radiation = lw_scheme)).model.time_stepping
    steps_per_day = steps_from_days(1, Δt)

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

        # Check every start day is a stored restart point
        all(d -> d % ref.full_gap == 0, start_days) || error(
            "Reference '$reference' stores restart states only every $(ref.full_gap) days, " *
            "so it cannot be initialized at all of $(start_days)."
        )

        # Learn the shape of every probed field from the reference
        ncols = map(probe -> n_cols(probe(verify_state(ref, 0))), probes)

        # Prepare containers for metrics and global means
        curve(nc) = zeros(Float64, max_horizon, n_traj, nc)
        err  = map(nc -> (; rmse = curve(nc), bias = curve(nc)), ncols)
        glob = (; run = map(curve, ncols), ref = map(curve, ncols))

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

                # Verification fields of the reference at day s+h
                ref_state = verify_state(ref, s + h)

                # Loop over probed fields
                for (p, probe) in pairs(probes)

                    f_run = traj[p][h+1]            # rollout at lead day h
                    f_ref = probe(ref_state)        # reference at day s+h

                    # Loop over column entries (layers for profiles, one pass for scalars)
                    for k in 1:ncols[p]
                        x, y = get_col(f_run, k), get_col(f_ref, k)

                        # Calculate metrics
                        err[p].rmse[h,i,k] = wrmse(x, y, w)
                        err[p].bias[h,i,k] = wbias(x, y, w)

                        # Calculate global means (drift diagnostics)
                        glob.run[p][h,i,k] = wmean(x, w)
                        glob.ref[p][h,i,k] = wmean(y, w)
                    end
                end
            end

            # Keep the full fields of one trajectory for the heatmaps
            if i == heatmap_traj
                heatmap_states = map(fields -> [fields[d+1] for d in heatmap_days], traj)
                heatmap_ref    = map(probe -> [probe(verify_state(ref, s + d)) for d in heatmap_days], probes)
            end

            @info "Trajectory $i/$(n_traj) (start day $s) finished!"
        end


        # Collect rollout results
        return (;
            days            = 1:max_horizon,
            probes          = keys(probes),
            start_days      = start_days,

            err             = err,
            glob            = glob,

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
