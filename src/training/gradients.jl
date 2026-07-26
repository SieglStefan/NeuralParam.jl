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
    do_autodiff,
    grid_weights
)

    # To be trained parameterization used for multiple dispatch only
    para = sim_train.model.longwave_radiation


    # Start AD from sim_train.variables (carry nn_input etc.), reset state to initial variables
    vars_ad = deepcopy(sim_train.variables)
    copy!(vars_ad, vars0)

    # Define and seed gradient container for autodiff
    bvars_ad = make_zero(vars_ad)

    # Seed gradient container and calculate loss
    loss = seed_loss!(bvars_ad, para, sim_train.variables, sim_target.variables, grid_weights)

    # Compute metric losses
    metrics = compute_metrics(para, sim_train.variables, sim_target.variables)

    # Copy training model and seed model gradient container
    model_ad = deepcopy(sim_train.model)
    bmodel_ad = make_zero(model_ad)

    

    # In test mode, skip Enzyme compilation and return zero gradients
    if ~do_autodiff
        return bmodel_ad.longwave_radiation.ps, loss, metrics
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

    return grads, loss, metrics
end



# Perform several timestep! calls with checkpointing for reverse-mode AD
function checkpointed_timesteps!(
    vars_ad,
    model_ad,
    n_steps,
    checkpoint_scheme::Scheme,
)

    # Perform n_steps of timestep! with checkpointing for reverse-mode AD
    @ad_checkpoint checkpoint_scheme for _ in 1:n_steps
        SpeedyWeather.time_step!(vars_ad, model_ad.time_stepping, model_ad)             # propagate dynamics
        SpeedyWeather.time_step!(vars_ad.prognostic.clock, model_ad.time_stepping)      # propagate clock
    end

    return nothing
end