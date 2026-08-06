### Configuration files for training a parameterization
###
### Contains:
###     - RunConfig:    contains parameters for a training run
###     - OutputConfig: contains parameters for outputs of the training



# Struct holding training run configuration parameters
@kwdef struct RunConfig
    seed::Int = 1234                                    # seed for RNG
    name::String = "default"                            # name of the training run
    dir::String = "default"                             # directory for storing training runs
    
    model::Type = PrimitiveWetModel                                         # model used for training
    lw_target::Union{Nothing, SpeedyWeather.AbstractLongwave} = nothing     # target LW scheme

    eta0::Float32 = 1f-2                                # initial learning rate
    eta_decay::Float32 = 0.8f0                          # learning rate decay after an ic
    loss_config::Union{Nothing, LossConfig} = nothing   # weighting and normalization of the loss

    t_spinup::Period = Day(30)                          # spinup time before training
    start_date::DateTime = DateTime(2000, 1, 1)         # start date of simulation

    n_ic::Int = 5                       # nr. of ic used for training
    n_traj::Int = 100                   # nr. of trajectroies per ic u.f.t.
    n_batch::Int = 4                    # batch size for training
    n_steps_0::Int = 5                  # nr. of initial training steps per update
    n_steps_inc::Int = 2                # increase of n_steps after an ic
    n_gap::Int = 50                     # nr. of timestep!() between two trajectories

    fac_pert_T::Float32 = 2f0           # additive perturbation factor for temperature
    fac_pert_q::Float32 = 0.2f0         # multiplicative perturbation factor for humidity

    do_autodiff::Bool = true            # whether to use autodiff for training
end