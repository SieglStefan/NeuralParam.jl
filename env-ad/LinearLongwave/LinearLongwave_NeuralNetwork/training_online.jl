
using SpeedyWeather, Random, Enzyme

extract_gradients(::NeuralLinearLongwave, bmodel_rad) =
    bmodel_rad.longwave_radiation.ps



# Function for preparing data and autodiffing the loss function
function compute_gradients!(vars_0, vars_target, model_rad, dt)

    # Copy initial variables and propagate the radiant model
    vars_rad = deepcopy(vars_0)
    SpeedyWeather.timestep!(vars_rad, dt, model_rad)


    # Simplicity definitions
    T_target = vars_target.grid.temperature
    T_rad = vars_rad.grid.temperature
    N = length(T_rad)


    # Copy initial variables for autodiff and seed for loss function
    vars_ad = deepcopy(vars_0)
    bvars_ad = make_zero(vars_ad)
    bvars_ad.grid.temperature .= 2 .* (T_rad .- T_target) ./ N

    # Create model gradient container
    bmodel_rad = make_zero(model_rad)


    # Differentiate the loss function regarding to the model parameters (and variables)
    autodiff(Reverse,
            SpeedyWeather.timestep!, 
            Const,                                 
            Duplicated(vars_ad, bvars_ad),        # Mutable state propagated by model_rad
            Const(dt),                              # Time step for the timestep function
            Duplicated(model_rad, bmodel_rad))      # Model with parameters to differentiate
       

    # Extract gradients for the parameters a and b  
    grads = extract_gradients(model_rad.longwave_radiation, bmodel_rad)  
    

    # Calculate loss
    L = MSE(T_rad, T_target)

    return L, grads
end



function training_step(spectral_grid; neural_para,
                            sim_target, dt,
                            opt_state,
                            printing=true)


    model_rad = PrimitiveWetModel(; spectral_grid, longwave_radiation=neural_para)

    # Cache initial variables
    vars_0 = deepcopy(sim_target.variables)

    # Propagate the target model
    SpeedyWeather.timestep!(sim_target.variables, dt, sim_target.model)

    # Calculate gradients
    loss, grads = compute_gradients!(vars_0, sim_target.variables, model_rad, dt)

    # Update parameters and storage
    updates, opt_state = Optimisers.update(opt_state, neural_para.ps, grads)
    ps_new = Optimisers.apply_updates(neural_para.ps, updates)

    neural_para_new = NeuralLinearLongwave(;
        nn = neural_para.nn,
        ps = ps_new,
        st = neural_para.st,
        config = neural_para.config)

    if printing
        println("Loss = $loss")
    end

    return loss, neural_para_new, opt_state
end


function perturb_temp!(sim; A=1., rng=Random.default_rng())
    T_grid = copy(sim.variables.grid.temperature)
    noise = randn!(rng, similar(T_grid))
    T_grid .+= A .* noise

    set!(sim, temperature = T_grid)

    return nothing
end


function run_training(spectral_grid; 
                            eta=1f-4,
                            n_ic=10, n_steps=100, config, rng,
                            printing=true)


    # Create spectral grid 
    model0 = PrimitiveWetModel(; spectral_grid)

    # Extract timestepping
    (; Δt, Δt_millisec) = model0.time_stepping
    dt = 2Δt
    
    L = []

    radiation = NeuralLinearLongwave(spectral_grid; config, rng)

    rule = Optimisers.Adam(eta)
    opt_state = Optimisers.setup(rule, radiation.ps)

    for i in 1:n_ic

        sim = initialize!(model0)

        perturb_temp!(sim)
        
        run!(sim, period=Hour(24))

        

        for j in 1:n_steps

            loss, radiation, opt_state = training_step(spectral_grid;
                                                        neural_para = radiation,
                                                        sim_target = sim,
                                                        model_target,
                                                        dt)
                                                        


            push!(L, loss)
        end

        if printing
            println("Initial condition Nr. $i / $n_ic finished!, current loss: $loss")
        end
    end

    return L, radiation

end