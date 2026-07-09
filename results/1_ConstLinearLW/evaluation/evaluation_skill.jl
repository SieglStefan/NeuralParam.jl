using SpeedyWeather        # SpectralGrid, PrimitiveWetModel, OneBandLongwave, initialize!, run!, ...
using Dates                # Day
using Random               # Random.seed!
using Plots                # zum Plotten
using Statistics



function forecast_skill(spectral_grid, scheme_target, schemes::NamedTuple;
                        n_steps, n_segments, n_gap, fac_pert_T, fac_pert_q, t_spinup, seed)
    Random.seed!(seed)

    # Referenz: perturbieren + einlaufen + fürs Stepping vorbereiten
    sim_ref = initialize!(PrimitiveWetModel(spectral_grid))
    perturb_grid_field!(sim_ref, :temperature; fac_add = fac_pert_T)
    perturb_grid_field!(sim_ref, :humidity;    fac_mult = fac_pert_q, zeromin = true)
    run!(sim_ref, period = t_spinup)
    SpeedyWeather.initialize!(sim_ref, steps = n_segments*(n_steps+n_gap)+1)
    SpeedyWeather.first_timesteps!(sim_ref)

    # target-sim + je 1 sim pro scheme
    sim_target = initialize!(PrimitiveWetModel(spectral_grid; longwave_radiation = scheme_target))
    sims = map(scheme -> initialize!(PrimitiveWetModel(spectral_grid; longwave_radiation = scheme)), schemes)

    err_field = Dict{Symbol,Any}()                                  # Dict, weil veränderlich!
    rmse_vals = Dict(name => Float32[] for name in keys(schemes))
    bias_vals = Dict(name => Float32[] for name in keys(schemes))

    for seg in 1:n_segments
        vars0 = deepcopy(sim_ref.variables)

        copy!(sim_target.variables, vars0)
        NeuralParam.sim_timesteps!(sim_target, n_steps)             # nicht exportiert → qualifizieren
        Tt = sim_target.variables.grid.temperature

        for name in keys(schemes)
            copy!(sims[name].variables, vars0)
            NeuralParam.sim_timesteps!(sims[name], n_steps)
            Tc = sims[name].variables.grid.temperature
            if seg == 1
                err_field[name] = copy(Tc .- Tt)
            else
                err_field[name] .+= (Tc .- Tt)
            end
            push!(rmse_vals[name], rmse(Tc, Tt))
            push!(bias_vals[name], bias(Tc, Tt))
        end

        for _ in 1:(n_steps + n_gap); SpeedyWeather.later_timestep!(sim_ref); end
    end

    for name in keys(schemes); err_field[name] ./= n_segments; end  # mitteln
    return (; err_field, rmse_vals, bias_vals)
end


TRUNC = 31
NLAYERS = 8

SAMPLE_RES = 1


spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS)

run_const = "run_T31_L8_2026-07-07_23-22-44"
scheme_comp = load_scheme(
    path = joinpath(@__DIR__, "..", "calibration", run_const),
    file = "scheme.jld2"
)

model = PrimitiveWetModel(spectral_grid)

n_steps = steps_from_days(1, model.time_stepping.Δt_sec)

data = forecast_skill(
    spectral_grid,
    OneBandLongwave(spectral_grid),
    (; comp = scheme_comp,
        none = nothing,
        uncal = ConstLinearLW(spectral_grid)),
    n_steps = n_steps,
    n_segments = 365,
    n_gap = 10,
    fac_pert_T = 2f0,
    fac_pert_q = 0f0,
    t_spinup = Day(31),
    seed = 42
)

@show mean(data.rmse_vals[:comp]) mean(data.bias_vals[:comp])
@show mean(data.rmse_vals[:none]) mean(data.rmse_vals[:uncal])


histogram(data.rmse_vals[:comp]; bins=30, xlabel="RMSE (K)", ylabel="Segmente",
          title="1-day forecast RMSE — comp", legend=false)


plot_heatmaps(extract_layer(8, [data.err_field[:comp]]); titles=["comp — mean 1d error, layer 8"])


prof(f) = [Statistics.mean(abs.(f[:, k])) for k in 1:NLAYERS]   # mittlerer |Fehler| pro Schicht
plot(prof(data.err_field[:comp]), 1:NLAYERS; yflip=true, xlabel="mean |error| (K)", ylabel="Layer", marker=:circle)