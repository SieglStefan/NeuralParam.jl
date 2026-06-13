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
function compute_gradients(vars0, sim_target, sim_train, n_steps, test_mode)

    # Extract final temperature fields
    T_target = sim_target.variables.grid.temperature
    T_train = sim_train.variables.grid.temperature
    N = length(T_train)

    # Copy initial variables, so timestep! does not mutate vars0
    vars_ad = deepcopy(vars0)

    # RMSE is logged as loss value.
    # The AD seed below uses the MSE gradient, which has the same minimum.
    L = rmse(T_train, T_target)


    # Create shadow variables and seed reverse AD with dMSE/dT_train_out, where
    # T_out is the final temperature after n_steps.
    #
    # Before autodiff:
    #   bvars_ad.grid.temperature = dL/dT_train_out = 2 .* (T_train_out .- T_target_out) ./ N
    #          -> L = (T_train_out - T_target_out)^2 / N = MSE
    #
    # After autodiff:
    #   bvars_ad contains dL/d(vars_ad input)
    #   bmodel_ad contains dL/d(model_ad input)
    #
    bvars_ad = make_zero(vars_ad)
    bvars_ad.grid.temperature .= 2 .* (T_train .- T_target) ./ N

    # Create model shadow.
    # Gradients with respect to trainable model parameters accumulate here.
    model_ad = deepcopy(sim_train.model)
    bmodel_ad = make_zero(model_ad)


    # In test mode, skip Enzyme compilation and return zero gradients
    if test_mode
        grads = extract_gradients(model_ad.longwave_radiation, bmodel_ad)
        return L, grads
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
    grads = extract_gradients(model_ad.longwave_radiation, bmodel_ad)

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






# Extract gradients from a constant LinearLongwave parameterization
extract_gradients(::AbstractConstLongwave, bmodel_rad) = (;
    a = bmodel_rad.longwave_radiation.a,
    b = bmodel_rad.longwave_radiation.b,
)


# Extract gradients from a neural LinearLongwave parameterization
extract_gradients(::AbstractLuxLongwave, bmodel_rad) =
    bmodel_rad.longwave_radiation.ps