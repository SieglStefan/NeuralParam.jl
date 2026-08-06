### End-to-end calibration plumbing test with do_autodiff = false.
### Exercises: setup_simulations -> prepare_reference -> seed_loss! -> compute_gradients
###            -> Optimisers.update -> update_ps -> csv logging -> plots -> scheme save

using NeuralParam
using SpeedyWeather
using Dates

const STATS = "zscore_oblw_smoketest"
SG = SpectralGrid(trunc = 31, nlayers = 8)

target() = OneBandLongwave(SG; transmissivity = FriersonLongwaveTransmissivity(SG))

function try_calibration(name, scheme; loss_config = LossConfig())
    println("\n=========== ", name, "  ", loss_config, " ===========")
    out_dir = mktempdir()

    cfg = RunConfig(
        lw_scheme    = target(),
        do_autodiff  = false,
        loss_config  = loss_config,
        t_spinup     = Day(1),
        n_ic         = 1,
        n_traj       = 4,
        n_epochs     = 2,
        n_batch      = 2,
        n_steps_0    = 1,
        n_steps_inc  = 1,
        n_gap        = 1,
    )

    out = OutputConfig(
        printing_ic = false, printing_traj = false,
        output_path = out_dir,
        train_save  = true,
        scheme_save = true,
        plots_save  = true,
    )

    param, L, PN, GN = run_training(SG, scheme; run_config = cfg, output_config = out)

    println("  returned scheme:  ", nameof(typeof(param)))
    println("  loss   (n=", length(L), "): ", L)
    println("  all finite:       ", all(isfinite, L))
    println("  grad norms:       ", GN, "   (all zero expected: do_autodiff=false)")
    println("  param norms:      ", PN)

    df = NeuralParam.csv_read(; path = out_dir, file = out.train_file)
    println("  csv cols:         ", propertynames(df))
    println("  csv rows:         ", size(df, 1), "  (== length(L): ", size(df,1) == length(L), ")")

    println("  scheme.jld2:      ", isfile(joinpath(out_dir, out.scheme_file)))
    println("  plots:            ", isdir(joinpath(out_dir, out.plots_folder)) ?
            readdir(joinpath(out_dir, out.plots_folder)) : "MISSING")

    back = NeuralParam.load(; path = out_dir, file = out.scheme_file)
    println("  reloaded scheme:  ", nameof(typeof(back)))
    return L
end

L1 = try_calibration("ConstLW / LinearOutput", ConstLW(SG, LinearOutput(), STATS))
L2 = try_calibration("ConstLW / DirectOutput", ConstLW(SG, DirectOutput(), STATS))
L3 = try_calibration("NeuralLW / LinearOutput",
        NeuralLW(SG, IN_NLW_OBLW, MLPConfig(n_hidden = 1, width = 8), LinearOutput(), STATS))

# weights must actually scale the loss: zeroing the two flux terms must lower it
L4 = try_calibration("ConstLW / Linear (T only)", ConstLW(SG, LinearOutput(), STATS);
        loss_config = LossConfig(weights = (; T = 1f0, olw = 0f0, slwd = 0f0)))
println("\nweighting sanity: L(1,1,1)[1]=", L1[1], "  L(1,0,0)[1]=", L4[1],
        "   lower: ", L4[1] < L1[1])

# :mean normalization must construct and give a different (not NaN) loss
L5 = try_calibration("ConstLW / Linear (:mean norm)", ConstLW(SG, LinearOutput(), STATS);
        loss_config = LossConfig(normalization = :mean))
println("\nnormalization sanity: std=", L1[1], "  mean=", L5[1], "   differ: ", L1[1] != L5[1])

# a bad normalization symbol must be rejected at construction, not deep in the loss
try
    LossConfig(normalization = :sdt)
    println("\nvalidation: NOT rejected  <-- BUG")
catch e
    println("\nvalidation: rejected :sdt -> ", sprint(showerror, e)[1:min(end, 70)])
end

println("\nCALIBRATION_SMOKE_OK")
