### Function for starting a training
###
### Mainly wraps training_online!()



# Run a training for a longwave parameterization
function run_training(
    spectral_grid,               
    lw_radiation_train;
    lw_radiation_target = nothing,             
    run_config = RunConfig(),
    output_config = OutputConfig(),
    test_mode = false,
)

    # Decide saving path
    if isnothing(output_config.save_path)
        save_path = "$(nameof(typeof(lw_radiation_train)))_L$(spectral_grid.nlayers)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    else
        save_path = output_config.save_path
    end


    # Run offline optimization loop
    # - not implemented yet


    # Run online optimization loop
    param, L, PN, GN = training_online(;
        spectral_grid,
        lw_radiation_train,
        lw_radiation_target,
        run_config,
        output_config,
        save_path,
        test_mode,
    )


    # If true, save parameterization
    if output_config.param_save
        save(param; path=save_path, file=output_config.param_file)
    end

    return param, L, PN, GN
end