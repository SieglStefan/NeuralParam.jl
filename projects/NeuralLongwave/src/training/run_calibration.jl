### Functions for calibrating the LinearLongwave parameterization



# Function for performing one calibration step (Calculating gradients and updating a and b)
function calibration_step!( radiation_llw;
                            vars0, 
                            sim_target, 
                            sim_train,
                            dt,
                            eta,
                            opt_state,             # Not explicitely needed for calibration, but needed for training, so we keep it here for the online optimization loop
                            step_update,
                            printing_updates)

    # Compute gradients 
    loss, grads = compute_gradients(vars0, sim_target, sim_train, dt)
 
    # Calculate new parameters with gradient descent 
        # (multiplication with parameters at the end = normalizing learning rate eta)
    a_new = radiation_llw.a - eta * grads.a * radiation_llw.sc_grad_a
    b_new = radiation_llw.b - eta * grads.b * radiation_llw.sc_grad_b

    # Update radiation_llw parameter
    radiation_llw.a = a_new
    radiation_llw.b = b_new


    # Print information about calibration for debugging
    if printing_updates
        println("Step $step_update, Loss=$loss, a=$a_new, b=$b_new, ba=$(grads.a), bb=$(grads.b)")
    end

    return loss, grads, nothing
end



# Function for calibrating a LinearLongwave parameterization
function run_calibration!(  radiation_llw,              # to be calibrated LinearLongwave parameterization
                            spectral_grid;              # defines truncation and number of vertical layers  
                            eta0 = 1f-8,                 # learning rate
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

    push!(P, (radiation_llw.a, radiation_llw.b))

    
    # Run online optimization loop for calibrating radiation_llw
    run_online_optimization!(;  scheme_step! = calibration_step!,
                                radiation = radiation_llw,
                                spectral_grid,        
                                eta0,             
                                t_spinup,     
                                n_ic,
                                n_updates,                
                                n_gap,
                                n_steps,          
                                printing_ic,  
                                printing_updates,
                                opt_state = nothing,    # no optimizer state needed for calibration, only for training
                                L, P, G)

    return L, P, G
end
