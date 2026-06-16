

function run_training(
    lw_radiation,               # parameterization to be optimized
    spectral_grid;              # spectral grid used for model construction
    training_config = TrainingConfig(),
    target_lw_radiation = nothing,


    printing_ic = true,         # print after every IC
    printing_traj = true,       # print after every trajectory update
    printing_epochs = false,    # print after every epoch

    name = "run1",

    log = false,
    train_dir = ".",

    save_model = false,
    model_dir = ".",

    test_mode = false,          # skip Enzyme.autodiff if true
)




    # Possible: run first offline optimization to get good initial guess for online optimization


    # Warn when running without autodiff
    if test_mode
        @warn "Test mode is activated! Enzyme.autodiff is NOT used!"
    end


    # Run online optimization loop
    lw_radiation, L, PN, GN = training_online(;
        lw_radiation,
        spectral_grid,
        training_config,
        target_lw_radiation,
        printing_ic,
        printing_traj,
        printing_epochs,
        name,
        log,
        train_dir,
        test_mode
    )

    if save_model
        filepath = save_longwave(; path = model_dir, radiation = lw_radiation)
        @info "Parameterization saved at $filepath"
    end

    return lw_radiation, L, PN, GN
end