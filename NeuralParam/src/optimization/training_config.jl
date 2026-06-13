@kwdef struct TrainingConfig
    
    seed::Int = 1

    eta0::Float32 = 1f-3               
    eta_decay::Float32 = 1f0          
    eta_decay_steps::Int = 10      

    patience::Int = 20
    min_delta::Float32 = 1f-5

    t_spinup = Day(14)    

    n_ic::Int = 10              
    n_traj::Int = 100            
    n_epochs::Int = 50             
    n_gap::Int = 10                
    n_steps::Int = 1                

    amp_pert_T::Float32 = 2f0         
    amp_pert_q::Float32 = 0.1f0         

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