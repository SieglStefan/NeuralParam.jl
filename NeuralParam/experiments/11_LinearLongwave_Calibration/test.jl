# Function for creating, perturbing and spinup-ing the target and training simulation

using Random
using SpeedyWeather

function create_sim_pair(spectral_grid, sim_template; radiation, t_spinup)

    # Copy template simulation
    sim_pert = deepcopy(sim_template)

    # Perturbate the prepared simulation and spinup
    perturb_grid_temp!(sim_pert)
    run!(sim_pert, period=Hour(t_spinup))

    # Create the target simulation by copying the perturbated simulation
    sim_target = deepcopy(sim_pert)

    # Create the training model and create training simulation by copying
    model_train = PrimitiveWetModel(; spectral_grid, longwave_radiation = radiation)
    sim_train = initialize!(model_train)
    copy!(sim_train.variables, sim_pert.variables)

    return sim_target, sim_train
end

### Utility functions for pertubing ICs of simulations



# Function for pertubing the grid temperature field of a simulation for IC sampling using white noise
function perturb_grid_temp!(sim; amp=2., rng=Random.default_rng())

    # Initialize simulation (if not initialized yet, T_grid is empty)
    initialize!(sim)

    # Copy the grid temperature field
    T_grid = copy(sim.variables.grid.temperature)

    # Create white noise
    noise = randn!(rng, similar(T_grid))

    # Add noise to the grid temperature field
    T_grid .+= amp .* noise

    # Assigning the perturbated field to the simulation and initialize it (without initialize!, sim.prognostic is not actualized)
    set!(sim, temperature = T_grid)
    initialize!(sim)

    return nothing
end



spectral_grid = SpectralGrid()

model = PrimitiveWetModel(spectral_grid)

sim_template = initialize!(model)

radiation = LinearLongwave()
t_spinup = 0


sim_target, sim_train = create_sim_pair(spectral_grid, sim_template; radiation, t_spinup)



@show radiation === sim_train.model.longwave_radiation