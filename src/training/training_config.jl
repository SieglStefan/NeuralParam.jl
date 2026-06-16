@kwdef struct TrainingConfig
    
    seed::Int = 1

    eta0::Float32 = 1f-3               
    eta_decay::Float32 = 0.7f0                

    patience::Int = 2
    min_delta::Float32 = 1f-4

    t_spinup = Day(31)    

    n_ic::Int = 10              
    n_traj::Int = 50
    n_epochs::Int = 30
    n_steps::Int = 10             
    n_gap::Int = 20                
                   

    fac_pert_T::Float32 = 2f0         
    fac_pert_q::Float32 = 0.2f0         

end




function TrainingConfig(::Val{:test})
    
    return TrainingConfig(
        t_spinup = Day(1),
        n_ic = 1,
        n_traj = 1,
        n_epochs = 1,
        n_gap = 1,
        n_steps = 1
    )
end