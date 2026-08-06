### Generation of zscore statistics
###
### Objective: Sample input and target output fields from perturbed simulations and calculate
###     - input statistics per variable:                     T, q, p, lat, lf, sst, lst, Ts
###     - direct output statistics:                          dT, olw, slwd
###     - linear output statistics by per-column least squares: a, b (per layer), c, d, e, f, g



### Sampling

# Evaluate the target longwave scheme for all columns of the current state
#   - the temperature tendency is accumulated by the scheme, so it has to be zeroed first
#   - the flux fields are overwritten, so they can be read directly
function target_outputs!(dT, olw, slwd, vars, model)

    # Fetch the tendency step the scheme writes into
    lw   = model.longwave_radiation
    dTdt = SpeedyWeather.get_tendency_step(vars.tendencies.grid.temperature, model.time_stepping, lw)

    # Zero the temperature tendency to isolate the longwave contribution
    fill!(dTdt, 0)

    # Call the target scheme column by column
    for ij in axes(vars.grid.temperature, 1)
        SpeedyWeather.parameterization!(ij, vars, lw, model)
    end

    # Copy out tendencies and fluxes
    dT   .= dTdt
    olw  .= vars.parameterizations.outgoing_longwave
    slwd .= vars.parameterizations.surface_longwave_down

    return nothing
end


# Collect input and target output fields from n_ic perturbed simulations
function collect_samples(
    sim_temp,
    model;
    n_ic,
    t_spinup,
    sim_time,
    sample_gap,
    fac_pert_T,
    fac_pert_q,
)

    # Extract timestepping and grid dimensions
    (; Δt) = model.time_stepping
    npoints = size(sim_temp.variables.grid.temperature, 1)
    nlayers = size(sim_temp.variables.grid.temperature, 2)


    # Calculate number of timesteps and samples
    n_steps_total = round(Int, sim_time *3600 *24 /Δt) + 1
    n_gap         = round(Int, sample_gap *3600 *24 /Δt)
    n_per_ic      = n_steps_total ÷ n_gap
    n_samples     = n_ic * n_per_ic

    # Print information
    @info "Samples per IC: $n_per_ic, total: $n_samples"
    @info "Time between samples (days): $(n_gap * Δt /3600 /24)"



    # Declare empty template container (indexing for later statistics, e.g. linear regression for a and b)
    prof() = zeros(Float32, npoints, nlayers, n_samples)        # profile variables
    scal() = zeros(Float32, npoints, n_samples)                 # scalar variables

    # Declare container for the target output fields
    targets = (; dT = prof(), olw = scal(), slwd = scal())

    # Declare container for the input fields, sampled through the INPUTS registry itself,
    # so what is normalized here is exactly what a scheme reads at runtime
    n_in   = n_inputs(INPUTS, nlayers)
    layout = input_layout(INPUTS, nlayers)
    raw    = zeros(Float32, npoints, n_in, n_samples)

    # Scratch arrays for one target evaluation
    dT_k   = zeros(Float32, npoints, nlayers)
    olw_k  = zeros(Float32, npoints)
    slwd_k = zeros(Float32, npoints)

    # Scratch buffer for the inputs of one column
    X = zeros(Float32, n_in)



    ### Main loop: perturb, spin up, propagate and sample
    idx = 1

    for i in 1:n_ic

        # Create simulation
        sim = deepcopy(sim_temp)

        # Perturb temperature and humidity fields
        perturb_grid_field!(sim, :temperature; fac_add = fac_pert_T)
        perturb_grid_field!(sim, :humidity; fac_mult = fac_pert_q, zeromin = true)

        # Spinup model
        run!(sim, period = t_spinup)

        # Initialize simulation and do a first step
        first_steps!(sim; steps=n_steps_total)


        # Propagate the simulation and sample fields
        for sample in 1:n_per_ic

            # Shortcut variables
            vars = sim.variables
            lw   = model.longwave_radiation

            # Evaluate the target scheme for all columns
            target_outputs!(dT_k, olw_k, slwd_k, vars, model)

            # Store inputs and outputs
            for ij in 1:npoints

                # Inputs, filled by the same function the scheme uses
                fill_inputs!(X, INPUTS, ij, vars, model, lw)
                raw[ij,:,idx] .= X

                # Target outputs
                for k in 1:nlayers
                    targets.dT[ij,k,idx] = dT_k[ij,k]
                end

                targets.olw[ij,idx]  = olw_k[ij]
                targets.slwd[ij,idx] = slwd_k[ij]
            end

            # Propagate simulation
            sim_timesteps!(sim, n_gap)
            idx += 1
        end

        @info "IC Nr. $i finished!"
    end


    # Slice the flat input samples back into named fields, profiles keeping their layer axis
    inputs = map(r -> length(r) == 1 ? raw[:,first(r),:] : raw[:,r,:], layout)

    return (; inputs..., targets...)
end


# Pick a random subset of columns for regression diagnostics
function subsample(s, n_cols; rng = Random.default_rng())
    idx = randperm(rng, size(s.T, 1))[1:n_cols]

    return (; idx,
        T = s.T[idx,:,:], dT = s.dT[idx,:,:],
        Ts = s.Ts[idx,:], olw = s.olw[idx,:], slwd = s.slwd[idx,:]
    )
end



### Statistics

# Input statistics, one entry per registered input
#   - profiles get one mean/std per layer, scalars a single pair
function input_stats(s, spec = INPUTS)
    return (; (name => (last(entry) === :profile ? mean_std_layers(s[name]) : mean_std(s[name]))
               for (name, entry) in pairs(spec))...)
end



### Generation

# Generate zscore statistics for a target scheme and store them as stats.jld2
function generate_zscore(
    spectral_grid;
    name        = "default",                # name of the statistics
    dir         = stats_dir(name),          # output directory
    seed        = 1234,                     # seed for RNG

    model       = PrimitiveWetModel,        # model used
    lw_scheme   = nothing,                  # target longwave scheme the outputs are sampled from

    output_forms = (DirectOutput(), LinearOutput(), PlanckOutput()),    # output forms fitted, one group each

    t_spinup    = Day(30),                  # spinup time before sampling
    start_date  = DateTime(2000, 1, 1),     # start date of the simulation
    n_ic        = 1,                        # number of initial conditions
    sim_time    = 365,                      # sampling time per IC in days
    sample_gap  = 3.65,                     # days between two samples

    fac_pert_T  = 2f0,                      # additive perturbation amplitude for temperature
    fac_pert_q  = 0.2f0,                    # multiplicative perturbation amplitude for humidity

    n_sub       = 200,                      # number of columns stored for the regression plots
    do_plots    = true,                     # whether validation plots are created
)

    # Set seed for reproducability
    Random.seed!(seed)

    # Create model and initialize simulation
    sim_model = model(spectral_grid; longwave_radiation = lw_scheme)
    sim_temp  = initialize!(sim_model)

    # Set starting time for spinup
    clock_start = start_date - t_spinup
    SpeedyWeather.set!(sim_temp.variables.prognostic.clock; time = clock_start, start = clock_start)


    # Collect samples
    s = collect_samples(
        sim_temp, sim_model;
        n_ic       = n_ic,
        t_spinup   = t_spinup,
        sim_time   = sim_time,
        sample_gap = sample_gap,
        fac_pert_T = fac_pert_T,
        fac_pert_q = fac_pert_q,
    )


    # Fit the coefficients of every output form, one group per form
    outputs = (; (output_group(f) => output_stats(f, s) for f in output_forms)...)

    # Calculate statistics
    stats = (;
        inputs = input_stats(s),
        outputs...,

        # shape metadata, checked when a scheme loads these stats
        nlayers = spectral_grid.nlayers,
        trunc   = spectral_grid.trunc,
    )

    # Store a subsample of the raw data for the regression plots
    sub = subsample(s, n_sub)


    # Save stats and regression subsample
    save(stats; dir, file = "stats.jld2")
    save(sub;   dir, file = "regression_samples.jld2")
    @info "Statistics $(name) stored at $(dir)!"


    # Create validation plots
    do_plots && plot_zscore(s, stats, sub; dir = joinpath(dir, "plots"), output_forms)

    return (; dir, stats, sub, samples = s)
end
