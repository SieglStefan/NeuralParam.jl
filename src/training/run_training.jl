### Function for starting a training
###
### Mainly wraps training_online!() and possibly training_offline!() in the future



# Run a training for a longwave parameterization scheme
function run_training(
    spectral_grid,                      # spectral_grid of the model         
    lw_train,                           # longwave parameterization scheme to train 
    run_config,                         # run configuration (RunConfig)
)


    # Run offline optimization loop
    # - not implemented yet


    # Run online optimization loop
    lw_trained = training_online(;
        spectral_grid,
        lw_train,
        rc = run_config,
    )


    # Save scheme after training
    save(lw_trained; dir=run_config.dir, file="scheme.jld2")
    @info "Scheme $(run_config.name) stored at $(run_config.dir)!"


    return lw_trained
end