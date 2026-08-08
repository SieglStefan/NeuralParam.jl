### Z-score statistics generation
###
### Writes to data/stats/<name>/:
###     stats.jld2                  the statistics (below)
###     plots/                      histograms + regression fits
###
### Structure of `data = load_zscore(name)` — every leaf is (; mean, std):
###
###     data.inputs.T       profile -> Vector(nlayers),  e.g. data.inputs.T.mean
###     data.inputs.Ts      scalar  -> Vector(1),        e.g. data.inputs.Ts.std[1]
###     data.direct         .dT (nlayers), .olw, .slwd
###     data.linear         .a .b (nlayers), .c .d .e .f .g  - fitted on centered T
###                         .center.T (nlayers), .center.Ts (1) — subtract before applying
###     data.planck         same as .linear, fitted on T^4
###     data.nlayers, data.trunc
###
### - scalars are length-1 Vectors, not numbers -> index with [1]
### - always holds ALL inputs and ALL output groups; a scheme picks its subset



### Sampling

# Collect input and target output fields from n_ic perturbed simulations
function collect_samples(
    sim_temp;
    n_ic,
    t_spinup,
    sim_time,
    sample_gap,
    fac_pert_T,
    fac_pert_q,
)

    # Extract timestepping and grid dimensions
    (; Δt) = sim_temp.model.time_stepping
    npoints = size(sim_temp.variables.grid.temperature, 1)
    nlayers = size(sim_temp.variables.grid.temperature, 2)


    # Calculate number of timesteps and samples
    n_steps_total = steps_from_days(sim_time, Δt) + 1
    n_gap         = steps_from_days(sample_gap, Δt)
    n_per_ic      = n_steps_total ÷ n_gap
    n_samples     = n_ic * n_per_ic

    # Print information
    @info "Samples per IC: $n_per_ic, total: $n_samples"
    @info "Time between samples (days): $(n_gap * Δt /3600 /24)"



    # Declare container for the input fields, sampled through the INPUTS registry itself
    n_in   = n_inputs(INPUTS, nlayers)
    layout = input_layout(INPUTS, nlayers)
    raw    = zeros(Float32, npoints, n_in, n_samples)

    # Declare container for the target output fields
    targets = (; 
        dT   = zeros(Float32, npoints, nlayers, n_samples),     # profile
        olw  = zeros(Float32, npoints, n_samples),              # scalar
        slwd = zeros(Float32, npoints, n_samples),              # scalar 
    )


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
        first_steps!(sim; planned_steps=n_steps_total)


        # Propagate the simulation and sample fields
        for sample in 1:n_per_ic

            # Shortcut variables
            vars = sim.variables
            model = sim.model
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

            
            idx += 1

            # Propagate only if another sample follows — the last block would be discarded
            sample < n_per_ic && sim_timesteps!(sim, n_gap)

        end

        @info "IC Nr. $i finished!"
    end


    # Slice the flat input samples back into named fields, profiles keeping their layer axis
    inputs = map(r -> length(r) == 1 ? raw[:,first(r),:] : raw[:,r,:], layout)

    return (; inputs..., targets...)
end


# Helper function for evaluating the target longwave scheme for all columns of the current state
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




### Generation

# Generate zscore statistics for a target scheme and store them as stats.jld2
function generate_zscore(;
    spectral_grid,              # used spectral grid
    name,                       # name of the zscore statistics, stored in data/stats/<name>/
    dir,                        # directory for storing the statistics
    seed,                       # used seed for generating data

    model,                      # used model type, e.g. PrimitiveWetModel
    lw_scheme,                  # used longwave scheme, e.g. OneBandLongwave

    output_forms,               # used output forms, e.g. (LinearOutput(), PlanckOutput())

    t_spinup,                   # spinup time for each perturbed initial condition (days)
    start_date,                 # starting date of the spinup (DateTime)
    n_ic,                       # number of perturbed initial conditions to sample
    sim_time,                   # simulation time for each perturbed initial condition (days)
    sample_gap,                 # time between samples (days)

    fac_pert_T,                 # perturbation factor for temperature (additive)
    fac_pert_q,                 # perturbation factor for humidity (multiplicative)
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
    samples = collect_samples(
        sim_temp;
        n_ic       = n_ic,
        t_spinup   = t_spinup,
        sim_time   = sim_time,
        sample_gap = sample_gap,
        fac_pert_T = fac_pert_T,
        fac_pert_q = fac_pert_q,
    )


    # Fit the coefficients of every output form, one group per form
    outputs = (; (output_group(f) => output_stats(f, samples) for f in output_forms)...)

    # Calculate statistics
    stats = (;
        inputs = input_stats(samples),
        outputs...,

        # Raw field stds — used by the LOSS, independent of any output form
        fields = (; T = mean_std_layers(samples.T),
                    olw = mean_std(samples.olw),
                    slwd = mean_std(samples.slwd)),

        # shape metadata, checked when a scheme loads these stats
        nlayers = spectral_grid.nlayers,
        trunc   = spectral_grid.trunc,
    )


    # Save stats
    save(stats; dir, file = "stats.jld2")
    @info "Statistics $(name) stored at $(dir)!"


    # Create validation plots
    plot_zscore(samples, stats; dir = joinpath(dir, "plots"), output_forms)

    return (; dir, stats, samples)
end


# Helper function for calculating input statistics (mean and std) for a given set of samples
function input_stats(samples, spec = INPUTS)
    return (; (name => (last(entry) === :profile ? mean_std_layers(samples[name]) : mean_std(samples[name]))
               for (name, entry) in pairs(spec))...)
end