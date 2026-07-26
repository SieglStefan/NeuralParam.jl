### Function for starting a training
###
### Mainly wraps training_online!()



# Run a training for a longwave parameterization
function run_training(
    spectral_grid,               
    lw_radiation_train,;             
    run_config = RunConfig(),
    output_config = OutputConfig(),
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
        run_config,
        output_config,
        output_path,
    )


    # Save scheme after training
    if output_config.scheme_save
        save(param; path=output_path, file=output_config.scheme_file)
    end

    return param, L, PN, GN
end