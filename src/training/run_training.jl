### Function for starting a training
###
### Mainly wraps training_online!()



# Run a training for a longwave parameterization
function run_training(
    spectral_grid,               
    lw_radiation_train,
    lw_radiation_target;             
    run_config = RunConfig(),
    output_config = OutputConfig(),
    test_mode = false,
)

    # Decide saving path
    if isnothing(output_config.output_path)
        output_folder = "$(nameof(typeof(lw_radiation_train)))_L$(spectral_grid.nlayers)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
        output_path = joinpath(pwd(), output_folder)
    else
        output_path = output_config.output_path
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
        output_path,
        test_mode,
    )


    # If true, save parameterization scheme
    if output_config.scheme_save
        save_scheme(param; path=output_path, file=output_config.scheme_file)
    end

    return param, L, PN, GN
end