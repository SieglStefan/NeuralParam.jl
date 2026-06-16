### Gradient computation with Enzyme
###
### This file contains the reverse-mode AD logic for one optimization step using Enzyme
###
### Idea:
### - sim_target and sim_train are propagated forward for n_steps.
### - The logged loss is RMSE between their final temperature fields.
### - The AD seed corresponds to the T_out MSE loss.
### - Enzyme backpropagates this loss and stores parameter gradients in bmodel_ad.




# Compute loss and gradients for one trajectory segment
function compute_gradients(
    vars0, 
    sim_target, 
    sim_train, 
    n_steps, 
    test_mode
)

    # To be trained parameterization used for multiple dispatch only
    para = sim_train.model.longwave_radiation


    # Copy initial variables, so timestep! does not mutate vars0
    vars_ad = deepcopy(vars0)

    # Define and seed gradient container for autodiff
    bvars_ad = make_zero(vars_ad)

    # Seed gradient container and calculate loss
    L = seed_loss!(bvars_ad, para, sim_train.variables, sim_target.variables)


    # Copy training model and seed model gradient container
    model_ad = deepcopy(sim_train.model)
    bmodel_ad = make_zero(model_ad)

    

    # In test mode, skip Enzyme compilation and return zero gradients
    if test_mode
        return L, bmodel_ad.longwave_radiation.ps
    end



    # Checkpointing avoids storing the full forward trajectory in memory
    checkpoint_scheme = Revolve(n_steps)

    # Differentiate n_steps of timestep! in reverse mode.
    Enzyme.autodiff(
        Enzyme.Reverse,
        checkpointed_timesteps!,
        Const,
        Duplicated(vars_ad, bvars_ad),
        Duplicated(model_ad, bmodel_ad),
        Const(n_steps),
        Const(checkpoint_scheme),
    )      

    # Extract parameter gradients from bmodel_ad
    grads = bmodel_ad.longwave_radiation.ps

    return L, grads
end



# Perform several timestep! calls with checkpointing for reverse-mode AD
function checkpointed_timesteps!(
    vars_ad,
    model_ad,
    n_steps,
    checkpoint_scheme::Scheme,
    lf1 = 2,
    lf2 = 2,
)
    @ad_checkpoint checkpoint_scheme for _ in 1:n_steps
        SpeedyWeather.timestep!(
            vars_ad,
            2 * model_ad.time_stepping.Δt,
            model_ad,
            lf1,
            lf2,
        )
    end

    return nothing
end




function seed_loss!(bvars_ad, ::AbstractLinearLW, vars_train, vars_target)
    

    # Extract final temperature fields
    T_train = vars_train.grid.temperature
    T_target = vars_target.grid.temperature
    N = length(T_target)


    # Seed reverse AD with dMSE/dT_train_out, where T_out is the final temperature after n_steps.
    #
    # Before autodiff:
    #   bvars_ad.grid.temperature = dL/dT_train_out = 2 .* (T_train_out .- T_target_out) ./ N
    #          -> L = (T_train_out - T_target_out)^2 / N = MSE
    #
    # After autodiff:
    #   bvars_ad contains dL/d(vars_ad input)
    #
    bvars_ad.grid.temperature.= 2f0 .* (T_train .- T_target)  ./ N

    return rmse(T_train, T_target)
end


# XXX  Change later
function seed_loss!(bvars_ad, ::AbstractABRLW, vars_train, vars_target)
    # Extract final temperature fields
    T_train = vars_train.grid.temperature
    T_target = vars_target.grid.temperature
    N = length(T_target)


    # Seed reverse AD with dMSE/dT_train_out, where T_out is the final temperature after n_steps.
    #
    # Before autodiff:
    #   bvars_ad.grid.temperature = dL/dT_train_out = 2 .* (T_train_out .- T_target_out) ./ N
    #          -> L = (T_train_out - T_target_out)^2 / N = MSE
    #
    # After autodiff:
    #   bvars_ad contains dL/d(vars_ad input)
    #
    bvars_ad.grid.temperature.= 2f0 .* (T_train .- T_target)  ./ N

    return rmse(T_train, T_target)
end