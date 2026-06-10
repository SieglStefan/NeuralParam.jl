### Main online optimization loop
###
### This file contains the main loop for calibrating/training longwave
### parameterizations against a target simulation.
###
### Workflow:
### 1. Create a template simulation that can be copied efficiently.
### 2. For each initial condition:
###    - Copy the template simulation.
###    - Randomly perturb its temperature field.
###    - Spin it up to obtain a physically reasonable training state.
### 3. From this spun-up state create three simulations:
###    - sim_ref: reference trajectory used to generate new trajectory segments.
###    - sim_target: target simulation with the default/true longwave scheme.
###    - sim_train: trainable simulation with the optimized longwave scheme.
### 4. For each trajectory segment:
###    - Advance sim_target to produce the target final state.
###    - Advance sim_train from the same initial state.
###    - Compute the loss between sim_train and sim_target.
###    - Backpropagate through sim_train and update the trainable parameters.
### 5. Move sim_ref forward and reset sim_target/sim_train to the next state.



# Run the full online optimization loop.
#
# The reference simulation provides fresh trajectory segments. On each segment,
# sim_target defines the target state, while sim_train is updated to match it.
function online_optimization!(; 
    radiation,
    spectral_grid,        
    eta0,             
    eta_fac,
    eta_steps,
    t_spinup,     
    n_ic,
    n_traj,
    n_epochs,                
    n_gap,
    n_steps,
    amp_pert_T,
    amp_pert_q,          
    printing_ic,  
    printing_traj,
    printing_epochs,
    test_mode,
    L, P, G, PN, GN,
)

    @info "Online optimization started!"

    # Copy initial learning rate
    eta = copy(eta0)

    # Setup optimiser
    rule = Optimisers.Adam(eta)
    opt_state = Optimisers.setup(rule, get_trainable_params(radiation))

    # Create template model and simulation and extract time stepping
    sim_template = create_template(spectral_grid)

    # Counts total optimization updates
    counter = 0

    # Loop over initial conditions
    for ic in 1:n_ic

        # Create reference, target, and training simulations out of the template simulation
        sim_ref, sim_target, sim_train = create_sims(
            spectral_grid, 
            sim_template;
            radiation, 
            t_spinup,
            amp_pert_T,
            amp_pert_q
        )

        # Initialize reference trajectory and do a first step
        SpeedyWeather.initialize!(sim_ref, steps = n_traj * (n_gap + n_steps) + 1)
        SpeedyWeather.first_timesteps!(sim_ref)
        

        #  Loop over trajectory segments
        for traj in 1:n_traj

            # Prepare target/train simulation pair
            vars0 = prepare_sim_pair!(sim_target, sim_train, n_steps)

            # Propagate target simulation for AD seed / loss
            sim_timesteps!(sim_target, n_steps)            


            # Reuse same trajectory segment for several updates
            for epoch in 1:n_epochs

                # Reset training simulation to trajectory start
                reset_sim!(sim_train, vars0)

                # Propagate training simulation with current parameters
                sim_timesteps!(sim_train, n_steps)

                if counter == 0
                    @info "Start 1st optmization step!"
                end

                # Perform one optimization step
                loss, grads, ps_new, grads_opt, opt_state = online_optimization_step!(
                    radiation;
                    vars0,
                    sim_target,
                    sim_train,
                    n_steps,
                    opt_state,
                    epoch,
                    printing_epochs,
                    test_mode,
                )                        


                # Store loss, parameters, gradients, and norms
                push!(L, loss)
                push!(P, extract_parameters(radiation))
                push!(G, deepcopy(grads))
                push!(PN, Float32(tree_l2norm(ps_new)))
                push!(GN, Float32(tree_l2norm(grads_opt)))
                    
                # Update counter
                counter += 1
            end


            # Print trajectory update
            if printing_traj
                print_traj(traj, L[end])
            end

            # Update learning rate
            if counter % eta_steps == 0
                eta *= eta_fac
                Optimisers.adjust!(opt_state, eta)
            end


            # Propagate reference trajectory forward
            sim_timesteps!(sim_ref, n_gap + n_steps)

            # Reset target and training simulations to new reference state
            reset_sim!(sim_target, sim_ref.variables)
            reset_sim!(sim_train, sim_ref.variables)
        end

        # Print IC update
        if printing_ic
            print_ic(ic, L[end])
        end
    end 
    
    println("Optimization finished! Total number of update steps: $counter")
end



# Function for performing one optimization step
function online_optimization_step!(
    radiation;
    vars0,
    sim_target,
    sim_train,
    n_steps,
    opt_state,
    epoch,
    printing_epochs,
    test_mode,
)

    # Compute gradients 
    loss, grads = compute_gradients(vars0, sim_target, sim_train, n_steps, test_mode)

    # Extract trainable parameters and optimizer gradients
    ps = get_trainable_params(radiation)
    grads_opt = get_trainable_grads(radiation, grads)

    # Update parameters
    opt_state, ps_new = Optimisers.update(opt_state, ps, grads_opt)
    set_trainable_params!(radiation, ps_new)


    # Print epoch update
    if printing_epochs
        print_epochs(radiation, epoch, loss, grads_opt)
    end

    return loss, grads, ps_new, grads_opt, opt_state
end






# Helper function for getting trainable parameters for Optimisers
get_trainable_params(r::AbstractConstLongwave) = (;
    a = Float32[r.a / r.sc_a],
    b = Float32[r.b / r.sc_b],
)

# Helper function for getting trainable parameters for Optimisers
get_trainable_params(r::AbstractLuxLongwave) = r.ps



# Helper function for extracting parameters from a constant LinearLongwave parameterization
function extract_parameters(radiation::AbstractConstLongwave)
    return (radiation.a, radiation.b)
end

# Helper function for extracting parameters from a neural LinearLongwave parameterization
function extract_parameters(radiation::AbstractLuxLongwave)
    return deepcopy(radiation.ps)
end



# Helper function for extracting and scaling gradients
get_trainable_grads(r::AbstractConstLongwave, g) = (;
    a = Float32[g.a * r.sc_a],
    b = Float32[g.b * r.sc_b],
)

# Helper function for extracting neural gradients
get_trainable_grads(r::AbstractLuxLongwave, g) = g



# Helper function for setting parameters of the constant LinearLongwave parameterization
function set_trainable_params!(r::AbstractConstLongwave, ps)
    r.a = ps.a[1] * r.sc_a
    r.b = ps.b[1] * r.sc_b

    return nothing
end

# Helper function for setting parameters of a neural LinearLongwave parameterization
function set_trainable_params!(r::AbstractLuxLongwave, ps)
    r.ps = ps

    return nothing
end