### Trains a parameterization online in SpeedyWeather.jl
###
### Algorithm:
###     
###     - Setup optimiser and simulations
###             - template: sim used for copying before perturbation
###             - target:   sim used as target for gradient computation
###             - train:    sim of to be trained lw scheme
###     - Prepare containers and logging/saving 
###
###     - Loop over initial conditions: n_ic
###         - Copy sim_template and perturb it:             sim_pert
###         - Spinup sim_pert
###         - Copy sim_pert for reference:                  sim_ref
###
###         - Loop over trajectories: n_traj
###             - Copy sim_ref onto sim_target and sim_train
###             - Propagate sim_target and sim_train for n_steps
###             - Calculate loss and gradients
###             - Accumulate gradients over batch window
###             - (Update optimiser and scheme after batch window)
###             - Store training data
###             - Propagate sim_ref for n_gap steps
###
###         - Update learning rate



# Function for running a online training
function training_online(;
    spectral_grid,      # spectral_grid of the model     
    lw_train,           # longwave parameterization scheme to train
    rc,                 # run configuration (RunConfig)
)

    # Set seed for reproducability
    Random.seed!(rc.seed)


    # Setup optimiser
    opt_state, eta = setup_optimiser(rc, ps=lw_train.ps)

    # Setup simulations (template, training and target)
    sims = setup_simulations(spectral_grid, rc, lw_train)


    # Initalize .csv file for logging
    metric_keys = keys(compute_metrics(lw_train, rc, sims, make_zero(lw_train.ps)))
    csv_init(metric_keys; dir=rc.dir, file="training.csv")

    # Initialize folder for training plots
    mkpath(joinpath(rc.dir, "train_plots"))


    # Shuffle ics
    bin_order = randperm(rc.n_ic)


    # Print training start information
    @info "Online training started!"

    if !rc.do_autodiff
        @warn "Autodiff is deactivated! Enzyme.autodiff is NOT used!"
    end

    print_config(rc, sims.template.model.time_stepping.Δt)

    

    ### Main training loop
    # Loop over initial conditions
    for ic in 1:rc.n_ic

        # Update number of steps used for calculating gradients
        n_steps = rc.n_steps_0 + (ic-1) * rc.n_steps_inc

        # Draw a starting date
        start_date = sample_start_date(bin_order[ic], rc.n_ic)
    
        # Prepare reference simulation 
        sim_ref = prepare_reference(sims.template, rc, n_steps, start_date)


        # Declare gradient sum and update flag for training step
        grad_sum = nothing
        do_update = false



        # Loop over trajectory segments
        for traj in 1:rc.n_traj

            # Copy reference variables
            vars0 = deepcopy(sim_ref.variables) 

            # Set target variables to reference variables
            copy!(sims.target.variables, vars0)
            # Propagate target simulation for gradient computation
            sim_timesteps!(sims.target, n_steps)

            # Set training variables to reference variables
            copy!(sims.train.variables, vars0)
            # Propagate training simulation for gradient computation
            sim_timesteps!(sims.train, n_steps)


            # Update flag for training step
            if traj % rc.n_batch == 0
                do_update = true
            end

    
            # Print information of starting first training step
            if ic == 1 && traj== 1
                @info "Start 1st training step!"
            end

            # Perform one online gradient step
            step = online_gradient_step(;
                lw_train,
                rc,
                sims,
                vars0,
                n_steps,
                opt_state,
                grad_sum,
                do_update,
            )            


            # Compute metrics for logging
            metrics = compute_metrics(
                lw_train,
                rc,
                sims,
                step.grads,
            )


            # Extract gradient sum and optimser state
            grad_sum = step.grad_sum
            opt_state = step.opt_state

            # Update radiation scheme parameters after batch window
            if do_update
                lw_train = step.lw_train_updated
                sims = @set sims.train.model.longwave_radiation = lw_train
                do_update = false
            end


            # Write to .csv
            csv_row!(
                ic, traj, n_steps, eta;
                metrics = metrics,
                dir=rc.dir, file="training.csv"
            )


            # Propagate reference trajectory forward
            sim_timesteps!(sim_ref, n_steps+rc.n_gap)
        end


        # Update learning rate after every ic and update optimiser
        eta *= rc.eta_decay
        Optimisers.adjust!(opt_state, eta)


        # Plot current loss trajectory after every ic
        # Create plots
        p = plot_training(;
            dir = rc.dir, file = "training.csv", n_batch = rc.n_batch,
            plot_kwargs = (; plot_title = "until IC nr. $(ic) / $(rc.n_ic)")
        )
        pn = plot_metrics_norm(;
            dir = rc.dir, file = "training.csv", weights = rc.loss_config.weights,
            plot_kwargs = (; plot_title = "until IC nr. $(ic) / $(rc.n_ic)")
        )
        pr = plot_metrics_raw(;
            dir = rc.dir, file = "training.csv", weights = rc.loss_config.weights,
            plot_kwargs = (; plot_title = "until IC nr. $(ic) / $(rc.n_ic)")
        )

        # Prepare plots directory
        dir = joinpath(rc.dir, "train_plots")

        # Save plots
        tag = "IC_" * lpad(ic, 2, '0')
        Plots.savefig(p,  joinpath(dir, "training_$tag.png"))
        Plots.savefig(pn, joinpath(dir, "metrics_norm_$tag.png"))
        Plots.savefig(pr, joinpath(dir, "metrics_raw_$tag.png"))


        @info "Initial condition $(ic) / $(rc.n_ic) finished!"
    end

    @info "Training finished!"

    return lw_train
end



# Function for performing one gradient step
function online_gradient_step(;
    lw_train,
    rc,
    sims,
    vars0,
    n_steps,
    opt_state,
    grad_sum,
    do_update,
)

    # Compute gradients
    grads = compute_gradients(
        rc,
        sims,
        vars0,
        n_steps,
    )

    # Accumulate gradients over batch window
    grad_sum = isnothing(grad_sum) ? grads : tree_add(grad_sum, grads)


    # Keep the current scheme unless an update is due below
    lw_train_updated = lw_train

    # Calculate mean gradient when scheme update is due
    if do_update

        # Calculate mean
        grad_mean = tree_scale(grad_sum, 1f0/rc.n_batch)

        # Update optimiser and scheme
        opt_state, ps_new = Optimisers.update(opt_state, lw_train.ps, grad_mean)
        lw_train_updated = update_ps(lw_train, ps_new)

        # Reset gradient sum
        grad_sum = nothing
    end

    return (; lw_train_updated, grads, grad_sum, opt_state)
end