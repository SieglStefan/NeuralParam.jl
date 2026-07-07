### XXX
###
### XXX


function seed_loss!(bvars_ad, para::AbstractLinearLW, vars_train, vars_target)
    

    # Extract final temperature fields
    T_train = vars_train.grid.temperature
    T_target = vars_target.grid.temperature

    bvars_ad.grid.temperature.= 2f0 .* (T_train .- T_target) ./length(T_target)

    return mse(T_train, T_target)
end


# XXX 
function seed_loss!(bvars_ad, para::AbstractABRLW, vars_train, vars_target)

    # Extract number of vertical layers
    nlayers = para.n_out - 2

    # Extract final temperature fields
    T_train = vars_train.grid.temperature
    T_target = vars_target.grid.temperature

    # Extract fluxes
    olw_train = vars_train.parameterizations.outgoing_longwave
    olw_target = vars_target.parameterizations.outgoing_longwave

    slwd_train = vars_train.parameterizations.surface_longwave_down
    slwd_target = vars_target.parameterizations.surface_longwave_down


    T_std = reshape(para.zscore.input_std[1:nlayers], 1, :)  
    bvars_ad.grid.temperature  .=                       2f0 .* (T_train .- T_target)         ./T_std.^2      ./length(T_target)

    olw_std = para.zscore.output_std[end-1]
    bvars_ad.parameterizations.outgoing_longwave .=     2f0 .* (olw_train .- olw_target)     ./olw_std.^2    ./length(olw_target)

    slwd_std = para.zscore.output_std[end]
    bvars_ad.parameterizations.surface_longwave_down .= 2f0 .* (slwd_train .- slwd_target)   ./slwd_std.^2   ./length(slwd_target)

    return norm_mse(T_train, T_target, T_std) + norm_mse(olw_train, olw_target, olw_std) + norm_mse(slwd_train, slwd_target, slwd_std)
end


function compute_metrics(::AbstractABRLW, vars_train, vars_target)
    T_train = vars_train.grid.temperature
    T_target = vars_target.grid.temperature

    olw_train = vars_train.parameterizations.outgoing_longwave
    olw_target = vars_target.parameterizations.outgoing_longwave

    slwd_train = vars_train.parameterizations.surface_longwave_down
    slwd_target = vars_target.parameterizations.surface_longwave_down

    return(;
        rmse_T =    rmse(T_train, T_target),        bias_T =    bias(T_train, T_target),
        rmse_olw =  rmse(olw_train, olw_target),    bias_olw =  bias(olw_train, olw_target),
        rmse_slwd = rmse(slwd_train, slwd_target),  bias_slwd = bias(slwd_train, slwd_target),
    )
end


function compute_metrics(::AbstractLinearLW, vars_train, vars_target)
    T_train = vars_train.grid.temperature
    T_target = vars_target.grid.temperature

    return(;
        rmse_T =    rmse(T_train, T_target),        bias_T =    bias(T_train, T_target),
    )
end










    # Seed reverse AD with dMSE/dT_train_out, where T_out is the final temperature after n_steps.
    #
    # Before autodiff:
    #   bvars_ad.grid.temperature = dL/dT_train_out = 2 .* (T_train_out .- T_target_out) ./ N
    #          -> L = (T_train_out - T_target_out)^2 / N = MSE
    #
    # After autodiff:
    #   bvars_ad contains dL/d(vars_ad input)
    #



        # Seed reverse AD with dMSE/dT_train_out, where T_out is the final temperature after n_steps.
    #
    # Before autodiff:
    #   bvars_ad.grid.temperature = dL/dT_train_out = 2 .* (T_train_out .- T_target_out) ./ N
    #          -> L = (T_train_out - T_target_out)^2 / N = MSE
    #
    # After autodiff:
    #   bvars_ad contains dL/d(vars_ad input)
    #