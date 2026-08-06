### Shared functions for calculating zscore statistics
###
### Objective:
### - sampling input and target output fields from perturbed simulations
### - calculating input statistics per variable
### - calculating direct output statistics (dT, olw, slwd)
### - calculating linear output statistics (a, b, c, d, e, f, g) by per-column least squares



### Sampling functions

# Collect input and target output fields from n_ic perturbed simulations
function collect_samples(
    sim_temp,
    model,
    target_outputs!;
    n_ic,
    t_spinup,
    sim_time,
    sample_gap,
    fac_pert_T,
    fac_pert_q,
)

    # Extract timestepping and grid dimensions
    ts = model.time_stepping
    (; Δt) = ts
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

    # Declare container for input and target output fields
    s = (;
        # inputs
            T = prof(), q = prof(),
            p = scal(), lat = scal(), lf = scal(),
            sst = scal(), lst = scal(), Ts = scal(),
        # target outputs
            dT = prof(), olw = scal(), slwd = scal(),
    )

    # Scratch arrays for one target evaluation
    dT_k   = zeros(Float32, npoints, nlayers)
    olw_k  = zeros(Float32, npoints)
    slwd_k = zeros(Float32, npoints)



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

            # Fetch variables
            lw    = model.longwave_radiation
            T_g   = SpeedyWeather.get_prognostic_step(vars.grid.temperature, model.time_stepping, lw)
            q_g   = SpeedyWeather.get_prognostic_step(vars.grid.humidity, model.time_stepping, lw)
            sst_g = SpeedyWeather.get_prognostic_step(vars.prognostic.ocean.sea_surface_temperature, model.time_stepping, lw)
            ps_g  = vars.parameterizations.surface_pressure
            lf_g  = model.land_sea_mask.land_fraction


            # Evaluate the target scheme for all columns
            target_outputs!(dT_k, olw_k, slwd_k, vars, model)

            # Store inputs and outputs
            for ij in 1:npoints

                # Profile variables
                for k in 1:nlayers
                    s.T[ij,k,idx]  = T_g[ij,k]
                    s.q[ij,k,idx]  = log10(q_g[ij,k] + 1f-9)
                    s.dT[ij,k,idx] = dT_k[ij,k]
                end

                # Scalar variables
                s.p[ij,idx]   = ps_g[ij]
                s.lat[ij,idx] = model.geometry.latds[ij]
                s.lf[ij,idx]  = lf_g[ij]
                s.sst[ij,idx] = sst_g[ij]
                s.lst[ij,idx] = vars.prognostic.land.soil_temperature[ij,1]
                s.Ts[ij,idx]  = NeuralParam.surface_temp(ij, vars, model, lw)

                s.olw[ij,idx]  = olw_k[ij]
                s.slwd[ij,idx] = slwd_k[ij]
            end

            # Propagate simulation
            sim_timesteps!(sim, n_gap)
            idx += 1
        end

        println("\t\tIC Nr. $i finished!")
    end

    return s
end


# Pick a random subset of columns for regression diagnostics
function subsample(s, n_cols; rng = Random.default_rng())
    idx = randperm(rng, size(s.T, 1))[1:n_cols]
    
    return (; idx,
        T = s.T[idx,:,:], dT = s.dT[idx,:,:],
        Ts = s.Ts[idx,:], olw = s.olw[idx,:], slwd = s.slwd[idx,:]
    )
end



### Least squares fit functions

# Fit y = a + b*x for one column
function fit_linear(x::AbstractVector, y::AbstractVector)
    x_bar = mean(x)
    y_bar = mean(y)

    Sxx = sum(abs2, x .- x_bar)
    b = Sxx > 0 ? sum((x .- x_bar) .* (y .- y_bar)) / Sxx : 0f0

    return Float32(y_bar - b*x_bar), Float32(b)     # intercept (a), slope (b)
end

# Fit y = a + b*x1 + c*x2 for one column
function fit_linear(x1::AbstractVector, x2::AbstractVector, y::AbstractVector)
    A = hcat(ones(Float32, length(y)), Float32.(x1), Float32.(x2))
    c = A \ Float32.(y)
    return c[1], c[2], c[3]                         # a, b, c
end



### Statistics helpers

# Mean and std over all entries
mean_std(x) = (; mean = Float32[mean(x)], std = Float32[std(x)])

# Mean and std per vertical layer (averaging over all other dimensions)
mean_std_layers(x) = (;
    mean = Float32[mean(selectdim(x, 2, k)) for k in axes(x, 2)],
    std  = Float32[std(selectdim(x, 2, k))  for k in axes(x, 2)],
)



### Statistics

# Input statistics, one entry per available input variable
function input_stats(s)
    return (;
        T   = mean_std_layers(s.T),
        q   = mean_std_layers(s.q),
        p   = mean_std(s.p),
        lat = mean_std(s.lat),
        lf  = mean_std(s.lf),
        sst = mean_std(filter(isfinite, s.sst)),
        lst = mean_std(filter(isfinite, s.lst)),
        Ts  = mean_std(s.Ts),
    )
end


# Direct output statistics (DirectOutput: dT, olw, slwd)
function direct_stats(s)
    return (;
        dT   = mean_std_layers(s.dT),
        olw  = mean_std(s.olw),
        slwd = mean_std(s.slwd),
    )
end


# Linear output statistics (LinearOutput: dT = a*T + b, olw = c + d*T[1] + e*Ts,
#                           slwd = f + g*T[nlayers]), fitted per column
function linear_stats(s)

    npoints, nlayers, _ = size(s.T)

    # Temperature tendency coefficients, one fit per column and layer
    a = zeros(Float32, npoints, nlayers)
    b = zeros(Float32, npoints, nlayers)

    for ij in 1:npoints, k in 1:nlayers
        b[ij,k], a[ij,k] = fit_linear(view(s.T, ij, k, :), view(s.dT, ij, k, :))
    end

    # Flux coefficients, one fit per column
    c = zeros(Float32, npoints); d = zeros(Float32, npoints); e = zeros(Float32, npoints)
    f = zeros(Float32, npoints); g = zeros(Float32, npoints)

    for ij in 1:npoints
        c[ij], d[ij], e[ij] = fit_linear(view(s.T, ij, 4, :), view(s.Ts, ij, :), view(s.olw, ij, :))
        f[ij], g[ij]        = fit_linear(view(s.T, ij, nlayers, :), view(s.slwd, ij, :))
    end

    return (;
        a = mean_std_layers(a), b = mean_std_layers(b),
        c = mean_std(c), d = mean_std(d), e = mean_std(e),
        f = mean_std(f), g = mean_std(g),
    )
end


