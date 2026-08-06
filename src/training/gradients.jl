### Gradient computation with Enzyme
###
### Calculates the n_steps trajcetory of a SW simulation using Enzyme



# Gradient computation wrapper allowing do_autodiff=false testmode
function compute_gradients(rc, sims, vars0, n_steps)

    # Test mode: return zero gradients, without Enzyme ever being reached
    if !rc.do_autodiff
        return make_zero(sims.train.model.longwave_radiation.ps)
    end

    # Normal mode: call Enzyme.autodiff
    return Base.invokelatest(autodiff_gradients, rc, sims, vars0, n_steps)
end


# Gradient computation over a n_steps trajectory using Enzyme
@noinline function autodiff_gradients(rc, sims, vars0, n_steps)

    # Create adjoint vars from reference variables vars0
    vars_ad = deepcopy(sims.train.variables)
    copy!(vars_ad, vars0)

    # Calculate loss and seed AD with dL/dT
    bvars_ad = seed_loss!(rc, sims, vars_ad)


    # Copy training model and seed model gradient container with zeros
    model_ad = deepcopy(sims.train.model)
    bmodel_ad = make_zero(model_ad)


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

    return grads
end


# Perform several timestep! calls with checkpointing for reverse-mode AD
function checkpointed_timesteps!(
    vars_ad,
    model_ad,
    n_steps,
    checkpoint_scheme::Scheme,
)

    # Perform n_steps of time_step! with checkpointing for reverse-mode AD
    @ad_checkpoint checkpoint_scheme for _ in 1:n_steps
        SpeedyWeather.time_step!(vars_ad, model_ad.time_stepping, model_ad)             # propagate dynamics
        SpeedyWeather.time_step!(vars_ad.prognostic.clock, model_ad.time_stepping)      # propagate clock
    end

    return nothing
end
