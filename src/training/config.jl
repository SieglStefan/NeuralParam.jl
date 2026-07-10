### Configuration files for training a parameterization
###
### Contains:
###     - RunConfig:    contains parameters for a training run
###     - OutputConfig: contains parameters for outputs of the training



# Struct holdind training run configuration parameters
@kwdef struct RunConfig
    seed::Int = 1234                # seed for RNG

    eta0::Float32 = 1f-3            # initial learning rate       
    eta_decay::Float32 = 0.5f0      # learning rate decay after an ic         

    t_spinup = Day(31)              # spinup time before training

    n_ic::Int = 5                   # nr. of ic used for training
    n_traj::Int = 20                # nr. of trajectroies per ic u.f.t.
    n_epochs::Int = 5               # nr. of epochs per trajectory u.f.t.
    n_steps_0::Int = 20             # nr. of initial training steps per update
    n_steps_inc::Int = 10           # increase of n_steps after an ic
    n_gap::Int = 25                 # nr. of timestep!() between two trajectories
                   
    fac_pert_T::Float32 = 2f0       # additive perturbation factor for temperature
    fac_pert_q::Float32 = 0.2f0     # multiplicative perturbation factor for humidity
end


# Convenience constructor for a test run
function RunConfig(::Val{:test})
    
    return RunConfig(
        t_spinup = Day(1),
        n_ic = 2,
        n_traj = 2,
        n_epochs = 2,
        n_steps_0 = 1,
        n_steps_inc = 1,
        n_gap = 1,
    )
end



# Struct holding training output parameters
@kwdef struct OutputConfig
    printing_ic::Bool = true                    # print training info after every completed ic
    printing_traj::Bool = true                  # print -//- after every completed trajectory
    printing_epochs::Bool = false               # print -//- after every completed epoch

    output_path::Union{Nothing,String} = nothing  # training output saving folder

    train_save::Bool = true                     # save training information in a .csv
    train_file::String = "training.csv"         # file name

    scheme_save::Bool = true                    # save parameterization scheme after training
    scheme_file::String = "scheme.jld2"         # scheme file name

    plots_save::Bool = true                    # saves training plots after every ic
    plots_folder::String = "train_plots"        # folder where trainings plots are stored

end