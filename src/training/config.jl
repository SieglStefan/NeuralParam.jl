### Configuration files for training a parameterization
###
### Contains:
###     - RunConfig:    contains parameters for a training run
###     - OutputConfig: contains parameters for outputs of the training



# Struct holdind training run configuration parameters
@kwdef struct RunConfig
    eta0::Float32 = 1f-3            # initial learning rate       
    eta_decay::Float32 = 0.7f0      # learning rate decay after an ic         
    patience::Int = 2               # nr. of ic after training stops if no loss improvement
    min_delta::Float32 = 1f-4       # minimum loss difference for early training stop

    t_spinup = Day(31)              # spinup time before training

    n_ic::Int = 10                  # nr. of ic used for training
    n_traj::Int = 50                # nr. of trajectroies per ic u.f.t.
    n_epochs::Int = 30              # nr. of epochs per trajectory u.f.t.
    n_steps::Int = 10               # nr. of timestep!() for one training step
    n_gap::Int = 20                 # nr. of timestep!() between two trajectories
                   
    fac_pert_T::Float32 = 2f0       # additive perturbation factor for temperature
    fac_pert_q::Float32 = 0.2f0     # multiplicative perturbation factor for humidity
end


# Convenience constructor for a test run
function RunConfig(::Val{:test})
    
    return RunConfig(
        t_spinup = Day(1),
        n_ic = 1,
        n_traj = 1,
        n_epochs = 1,
        n_gap = 1,
        n_steps = 1
    )
end



# Struct holding training output parameters
@kwdef struct OutputConfig
    printing_ic::Bool = true                    # print training info after every completed ic
    printing_traj::Bool = true                  # print -//- after every completed trajectory
    printing_epochs::Bool = false               # print -//- after every completed epoch

    save_path::Union{Nothing,String} = nothing  # training output saving folder

    train_save::Bool = true                     # save training information in a .csv
    train_file::String = "training.csv"         # file name

    param_save::Bool = true                     # save parameterization after training
    param_file::String = "param.jld2"           # file name

    plotting::Bool = true                       # plot training after every ic
end