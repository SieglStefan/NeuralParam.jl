


function training_online(;
        lw_radiation,
        spectral_grid,
        training_config,
        target_lw_radiation,
        printing_ic,
        printing_traj,
        printing_epochs,
        name,
        log,
        train_dir,
        test_mode
    )

    (; eta0, eta_decay, patience, min_delta,
    t_spinup, n_ic, n_traj, n_epochs, n_gap, n_steps,
    fac_pert_T, fac_pert_q) = training_config


    # Containers for logging
    L = Float32[]       # loss
    PN = Float32[]      # parameter norm
    GN = Float32[]      # gradient norm


    # Setup optimiser
    eta = eta0

    rule = Optimisers.Adam(eta)
    opt_state = Optimisers.setup(rule, lw_radiation.ps)

    best_loss = 1000f0
    stale = 0
    best_ps = deepcopy(lw_radiation.ps)


    # Create template model and simulationfor later copying
    model_template = PrimitiveWetModel(; spectral_grid)
    sim_template = initialize!(model_template)


    # Create target and training simulation and do a first timestep for later unchanged model fields
    model_target = isnothing(target_lw_radiation) ?
        PrimitiveWetModel(; spectral_grid) :
        PrimitiveWetModel(; spectral_grid, longwave_radiation = target_lw_radiation)
    sim_target   = initialize!(model_target)
    SpeedyWeather.initialize!(sim_target, steps=0)
    SpeedyWeather.first_timesteps!(sim_target)

    model_train  = PrimitiveWetModel(; spectral_grid, longwave_radiation = lw_radiation)
    sim_train    = initialize!(model_train)
    SpeedyWeather.initialize!(sim_train, steps=0)
    SpeedyWeather.first_timesteps!(sim_train)


    # Print some information
    @info "Online training started!"
    print_config(training_config, 2*model_template.time_stepping.Δt_sec)
    
    # XXX
    if log
        path = joinpath(train_dir, "$name.csv")
        io = open(path, "w")

        println(io, "ic,traj,epoch,loss,eta,pnorm,gnorm") 
        flush(io)
    end

    # Loop over initial conditions
    for ic in 1:n_ic
    
        ### Prepare reference simulation

        # Copy template simulation for pertubation
        sim_pert = deepcopy(sim_template)

        # Perturb temperature and humidity fields
        perturb_grid_field!(sim_pert, :temperature, fac_add = fac_pert_T)
        perturb_grid_field!(sim_pert, :humidity, fac_mult = fac_pert_q, zeromin = true)
        
        # Spinup simulation 
        run!(sim_pert, period = Hour(t_spinup))

        # Create reference simulation
        sim_ref = deepcopy(sim_pert)

        # Initialize reference trajectory and do a first step
        SpeedyWeather.initialize!(sim_ref, steps = n_traj * (n_gap + n_steps) + 1)
        SpeedyWeather.first_timesteps!(sim_ref)



        # Loop over trajectory segments
        for traj in 1:n_traj

            # Copy reference variables
            vars0 = deepcopy(sim_ref.variables) 


            ### Prepare target simulation

            # Set target variables to reference variables
            copy!(sim_target.variables, vars0)

            # Propagate target simulation for gradient computation
            sim_timesteps!(sim_target, n_steps)



            # Reuse same trajectory segment for several updates
            for epoch in 1:n_epochs

                ### Prepare training simulation
            
                # Re-create training simulation with updated radiation
                sim_train = @set sim_train.model.longwave_radiation = lw_radiation

                # Set training variables to reference variables
                copy!(sim_train.variables, vars0)

                # Propagate training simulation for gradient computation
                sim_timesteps!(sim_train, n_steps)



                # Print information of starting first training step
                if ic == 1 && traj== 1 && epoch == 1
                    @info "Start 1st training step!"
                end

                # Perform one training step
                lw_radiation, loss, grads, ps_new, opt_state = online_training_step(
                    lw_radiation;
                    vars0,
                    sim_target,
                    sim_train,
                    n_steps,
                    opt_state,
                    test_mode
                )                        


                # Store loss, parameters, gradients, and norms
                push!(L, loss)
                push!(PN, Float32(tree_l2norm(ps_new)))
                push!(GN, Float32(tree_l2norm(grads)))

                if log
                    println(io, "$ic, $traj, $epoch, $(L[end]), $eta, $(PN[end]), $(GN[end])")
                    flush(io)
                end
                    

                # Print epoch update
                if printing_epochs
                    print_epochs(epoch, L[end], PN[end], GN[end])
                end
            end



            # Propagate reference trajectory forward
            for _ in 1:(n_steps+n_gap)
                SpeedyWeather.later_timestep!(sim_ref)
            end
            
            
            # Print trajectory update
            if printing_traj
                print_traj(traj, L[end], PN[end], GN[end])
            end
        end

        # Update learning rate
        eta *= eta_decay
        Optimisers.adjust!(opt_state, eta)

        if L[end] < best_loss - min_delta
            best_loss = L[end]
            best_ps = deepcopy(lw_radiation.ps)
            stale = 0
        else
            stale += 1
            if stale >= patience 
                @warn "Training finished early! (no loss decrease)"
                break
            end
        end

        # Print IC update
        if printing_ic
            print_ic(ic, L[end], PN[end], GN[end])
        end
    end 
    
    println("Training finished!")

    log && close(io)

    return update_ps(lw_radiation, best_ps), L, PN, GN
end



# Function for performing one training step
function online_training_step(
    lw_radiation;
    vars0,
    sim_target,
    sim_train,
    n_steps,
    opt_state,
    test_mode,
)

    # Compute gradients 
    loss, grads = compute_gradients(
        vars0, 
        sim_target, 
        sim_train, 
        n_steps, 
        test_mode
    )

    # Update parameters
    opt_state, ps_new = Optimisers.update(opt_state, lw_radiation.ps, grads)
    lw_radiation_new = update_ps(lw_radiation, ps_new)


    return lw_radiation_new, loss, grads, ps_new, opt_state
end


function update_ps(lw::NeuralLinearLW, ps_new)
    return NeuralLinearLW(lw.nn, ps_new, lw.st, lw.config, lw.input_buffer)
end


function update_ps(lw::ConstLinearLW, ps_new)
    return ConstLinearLW(ps_new, lw.config)
end

function update_ps(lw::NeuralABRLW, ps_new)
    return NeuralABRLW(lw.nn, ps_new, lw.st, lw.config, lw.input_buffer)
end









# Propagate a simulation for n_steps using the leapfrog timestep size
function sim_timesteps!(sim, n_steps)

    # Extract time stepping
    dt = 2 * sim.model.time_stepping.Δt

    # Propagate the simulation for n_steps * dt
    for _ in 1:n_steps
        SpeedyWeather.timestep!(sim.variables, dt, sim.model, 2, 2)
    end

    return nothing
end



