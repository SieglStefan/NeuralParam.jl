### Functions for calculating and seeding losses
###
### XXX



# Function for seeding the loss function (MSE) for a LinearLW scheme
function seed_loss!(bvars_ad, para::AbstractLinearLW, vars_train, vars_target, grid_weights)
    
    # Extract final temperature fields
    T_train = vars_train.grid.temperature
    T_target = vars_target.grid.temperature
    N_T = length(T_target)

    # Prepare grid area weights
    W = reshape(grid_weights, :, 1)

    # Seed the temperature gradient container
    bvars_ad.grid.temperature.= 2f0 .* W .* (T_train .- T_target) ./N_T

    # Return L = MSE
    return sum(W .* abs2.(T_train .- T_target)) /N_T
end


# Function for seeding the loss function (MSE) for an ABRLW scheme
function seed_loss!(bvars_ad, para::AbstractABRLW, vars_train, vars_target, grid_weights)

    # Extract final temperature fields
    T_train = vars_train.grid.temperature
    T_target = vars_target.grid.temperature
    N_T = length(T_target)


    # Extract fluxes
    olw_train = vars_train.parameterizations.outgoing_longwave
    olw_target = vars_target.parameterizations.outgoing_longwave
    N_olw = length(olw_target)

    slwd_train = vars_train.parameterizations.surface_longwave_down
    slwd_target = vars_target.parameterizations.surface_longwave_down
    N_slwd = length(slwd_target)


    # Prepare grid area weights and extract number of vertical layers
    W = reshape(grid_weights, :, 1)
    nlayers = para.n_out - 2

    
    # Extract temperature mean and seed scaled temperature gradient container
    T_mean = reshape(para.zscore.input_mean[1:nlayers], 1, :)  
    bvars_ad.grid.temperature  .=                       2f0 .* W .* (T_train .- T_target)         ./T_mean.^2      ./N_T

    # Extract flux means and seed scaled flux gradient containers
    olw_mean = para.zscore.output_mean[end-1]
    bvars_ad.parameterizations.outgoing_longwave .=     2f0 .* W .* (olw_train .- olw_target)     ./olw_mean.^2    ./N_olw

    slwd_mean = para.zscore.output_mean[end]
    bvars_ad.parameterizations.surface_longwave_down .= 2f0 .* W .* (slwd_train .- slwd_target)   ./slwd_mean.^2   ./N_slwd

    # Return L
    return  sum(W .* abs2.(T_train .- T_target) ./T_mean.^2 ./N_T) +
            sum(W .* abs2.(olw_train .- olw_target) ./olw_mean.^2 ./N_olw) +
            sum(W .* abs2.(slwd_train .- slwd_target) ./slwd_mean.^2 ./N_slwd) 

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