### Functions for calculating and seeding losses
###
### Contains:
###     - Function for calculating and seeding the loss function (MSE) for an AbstractLW scheme
###     - Function for calculating unweighted metrics (RMSE, bias) for simulations
###     - Struct and utilities for information about the loss (weighting)



# Function for seeding the loss function (MSE) for an AbstractLW scheme
function seed_loss!(rc, sims, vars_ad)

    # Create empty gradient container for autodiff
    bvars_ad = make_zero(vars_ad)


    # Extract final temperature fields
    T_train = SpeedyWeather.get_step(sims.train.variables.grid.temperature)
    T_target = SpeedyWeather.get_step(sims.target.variables.grid.temperature)
    N_T = length(T_target)

    # Extract fluxes
    olw_train = sims.train.variables.parameterizations.outgoing_longwave
    olw_target = sims.target.variables.parameterizations.outgoing_longwave
    N_olw = length(olw_target)

    slwd_train = sims.train.variables.parameterizations.surface_longwave_down
    slwd_target = sims.target.variables.parameterizations.surface_longwave_down
    N_slwd = length(slwd_target)


    # Unpack and prepare weights, field normalizations and grid weights
    lw = rc.loss_config.weights
    fw = rc.loss_config.field_norm
    gw = rc.loss_config.grid_weights


    # Seed the gradient containers with dL/d(output), evaluated at the state after n_steps
    #   - the seed must be the exact gradient of the loss returned below,
    #     so any change to the loss has to be mirrored here
    #   - after Enzyme.autodiff, bvars_ad holds dL/d(vars_ad at the start of the segment)
    SpeedyWeather.get_step(bvars_ad.grid.temperature) .=
        lw.T    .* 2f0 .* gw .* (T_train .- T_target)       ./fw.T.^2    ./N_T

    bvars_ad.parameterizations.outgoing_longwave .=
        lw.olw  .* 2f0 .* gw .* (olw_train .- olw_target)   ./fw.olw.^2  ./N_olw

    bvars_ad.parameterizations.surface_longwave_down .=
        lw.slwd .* 2f0 .* gw .* (slwd_train .- slwd_target) ./fw.slwd.^2 ./N_slwd


    return bvars_ad
end



# Struct holding loss configuration parameters
@kwdef struct LossConfig
    weights::NamedTuple = (; T = 1f0, olw = 1f0, slwd = 1f0)        # loss weighting factors
    field_norm::NamedTuple                                          # field normalization factors (zscore)
    grid_weights                                                    # grid area weights
end

# Convenvience constructor for LossConfig loading zscore stats from a file
function LossConfig(
    spectral_grid::SpectralGrid,
    zscore_name::String; 
    weights = (; T = 1f0, olw = 1f0, slwd = 1f0)
)

    # Extract number of vertical layers
    nlayers = spectral_grid.nlayers

    # Caclulate grid weights
    grid_weights = area_weights(spectral_grid)


    # Load field normalization from zscore file
    field_norm = load_field_norm(zscore_name)


    # Check if length of T field_norm matches number of layers
    if length(field_norm.T) != nlayers
        error("LossConfig field_norm.T length $(length(field_norm.T)) does not match number of layers $nlayers.")
    end

    # Check if all field_norm entries are positive (0 would lead to Inf in loss)
    if !all(all(>(0), v) for v in field_norm)
        error("LossConfig field_norm entries must all be positive.")
    end


    return LossConfig(weights, field_norm, grid_weights)
end


# Load field normalization factors from zscore file
function load_field_norm(zscore_name::String)

    data = load_zscore(zscore_name)


    return (;
        T    = reshape(Float32.(getproperty(data.inputs.T, :std)), 1, :),
        olw  = Float32(getproperty(data.direct.olw, :std)[1]),
        slwd = Float32(getproperty(data.direct.slwd, :std)[1]),
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