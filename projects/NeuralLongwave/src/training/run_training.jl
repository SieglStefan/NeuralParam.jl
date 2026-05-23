### Functions for training the NeuralLinearLongwave parameterization



# Function for performing one training step (Calculating gradients and updating NN)
function training_step!(radiation_nllw;
                        vars0, 
                        sim_target, 
                        sim_train,
                        dt,
                        eta,
                        opt_state,
                        step_update,
                        printing_updates)

    # Compute gradients 
    loss, grads = compute_gradients(vars0, sim_target, sim_train, dt)

    # Update radiation_nllw parameter
    opt_state, ps_new = Optimisers.update(opt_state, radiation_nllw.ps, grads)
    radiation_nllw.ps = ps_new


    # Print information about training for debugging
    if printing_updates
        println("Step $step_update, Loss=$loss")
        @show grads
    end

    return loss, grads, opt_state
end


# Function for training NeuralLinearLongwave parameterization online
function run_training!( radiation_nllw,              # to be calibrated LinearLongwave parameterization
                        spectral_grid;              # defines truncation and number of vertical layers  
                        eta0 = 1f-4,                 # learning rate
                        t_spinup = Day(14),          # spinup time for IC sampling
                        n_ic = 10,                  # number of IC used
                        n_updates = 100,            # number of updates per IC
                        n_gap = 10,                 # number of timesteps between 2 calibration points
                        n_steps = 1,                # number of steps used in autodiff (checkpointing) 
                        printing_ic = true,         # if true: loss is printed after every IC
                        printing_updates = true)    # if true: loss is printed after every calibration step

    
    # Create loss, parameters and gradients containers
    L = Float32[]
    P = []
    G = []

    push!(P, deepcopy(radiation_nllw.ps))


    # Setup the Optimisers for training
    rule = Optimisers.Adam(eta0)
    opt_state = Optimisers.setup(rule, radiation_nllw.ps)

    
    # Run online optimization loop for training radiation_nllw
    run_online_optimization!(   scheme_step! = training_step!,
                                radiation = radiation_nllw,
                                spectral_grid,        
                                eta0,             
                                t_spinup,     
                                n_ic,
                                n_updates,                
                                n_gap,
                                n_steps,          
                                printing_ic,  
                                printing_updates,
                                opt_state,
                                L, P, G)  

    return L, P, G
end
