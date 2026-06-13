

function run_training(;
    para,                       # parameterization to be optimized
    spectral_grid,              # spectral grid used for model construction
    training_config,


    printing_ic = true,         # print after every IC
    printing_traj = true,       # print after every trajectory update
    printing_epochs = false,    # print after every epoch

    test_mode = false,          # skip Enzyme.autodiff if true
)




    # Possible: run first offline optimization to get good initial guess for online optimization


    # Warn when running without autodiff
    if test_mode
        @warn "Test mode is activated! Enzyme.autodiff is NOT used!"
    end


    # Run online optimization loop
    L, P, G, PN, GN = training_online(;
        para,
        spectral_grid,
        training_config,
        printing_ic,
        printing_traj,
        printing_epochs,
        test_mode
    )

    return L, P, G, PN, GN
end