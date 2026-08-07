### Function for starting a training
###
### Mainly wraps training_online!() and possibly training_offline!() in the future



# Run a training for a longwave parameterization scheme
function run_training(
    spectral_grid,                      # spectral_grid of the model         
    lw_train,                           # longwave parameterization scheme to train 
    train_config,                       # train configuration (TrainConfig)
)


    # Run offline optimization loop
    # - not implemented yet


    # Run online optimization loop
    lw_trained = training_online(;
        spectral_grid,
        lw_train,
        tc = train_config,
    )


    # Save scheme after training
    save(lw_trained; dir=train_config.dir, file="scheme.jld2")
    @info "Scheme $(train_config.name) stored at $(train_config.dir)!"


    return lw_trained
end