


extract_gradients(::LinearLongwave, bmodel_rad) =
    (; a = bmodel_rad.longwave_radiation.a,
       b = bmodel_rad.longwave_radiation.b)

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



function calibration_step!(spectral_grid; step,
                            L, a, b,
                            vars_target, model_target, dt,
                            eta_a, eta_b,
                            printing=true)

    # Create LinearLongwave radiation model with current parameters
    radiation = LinearLongwave(spectral_grid; a=a[step], b=b[step])
    model_rad = PrimitiveWetModel(; spectral_grid, longwave_radiation=radiation)

    # Cache initial variables
    vars_0 = deepcopy(vars_target)

    # Propagate the target model
    SpeedyWeather.timestep!(vars_target, dt, model_target)

    # Calculate gradients
    loss, grads = compute_gradients!(vars_0, vars_target, model_rad, dt)
    ba = grads.a
    bb = grads.b

    # Update parameters and storage
    L[step] = loss
    a[step+1] = a[step] - eta_a * ba
    b[step+1] = b[step] - eta_b * bb

    if printing
        println("Step $step, Loss=$(L[step]), a=$(a[step]), b=$(b[step]), ba=$ba, bb=$bb")
    end
end



function run_calibration(spectral_grid; 
                            a0=-1f-6, b0=1f-3,
                            eta_a=1f-15, eta_b=1f-11, 
                            nsteps=100, ntime=10,
                            printing=true)

    # Create spectral grid and baseline model
    model_target = PrimitiveWetModel(; spectral_grid)

    # Initialize target model and spinup model
    simulation_target = initialize!(model_target)
    run!(simulation_target, period=Hour(12))

    # Extract target simulation variables
    vars_target = simulation_target.variables

    # Extract timestepping
    model_target = simulation_target.model
    (; Δt, Δt_millisec) = model_target.time_stepping
    dt = 2Δt


    # Define containers for loss and parameterization values
    L = zeros(Float32, nsteps)
    a = zeros(Float32, nsteps+1)
    b = zeros(Float32, nsteps+1)

    a[1] = a0
    b[1] = b0


    # Calibraiton loop
    for step in 1:nsteps

        # Do one step of calibration for a and b
        calibration_step!(spectral_grid; step,
                            L, a, b,
                            vars_target, model_target, dt,
                            eta_a, eta_b,
                            printing=printing)

        # Spinup model for more diverse data controlled by ntime
        for _ in 1:ntime
            SpeedyWeather.timestep!(vars_target, dt, model_target)
        end
    end

    return L, a, b

end