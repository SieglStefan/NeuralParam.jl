### Real Enzyme reverse-mode autodiff through SpeedyWeather's timestep!.
###
### SLOW: ≈1h COLD COMPILE *PER SCHEME*. Enzyme specializes on the model type, which
### embeds the scheme type — so a second scheme triggers a fresh compile; size of the
### grid/net barely matters (compile is type-driven). That is why this is gated
### per-scheme in runtests.jl (only re-test the scheme you changed).
###
### What it proves:
###   - loss stays finite,
###   - a real, NON-ZERO gradient was produced (Enzyme actually traversed the model,
###     incl. Lux.apply for the neural schemes),
###   - a gradient step did not INCREASE the loss — a flipped seed_loss! sign would.

@testset "autodiff (Enzyme)" begin

    # run a few epochs of REAL autodiff training on one fixed segment; return norms
    function autodiff_LPG(make_lw)
        lw  = make_lw(SG)
        cfg = RunConfig(n_ic = 1, n_traj = 1, n_epochs = 4, n_steps = 1, n_gap = 1, t_spinup = Day(1))
        out = OutputConfig(printing_ic = false, printing_traj = false,
                           save_path = mktempdir(), train_save = false,
                           param_save = false, plotting = false)
        _, L, PN, GN = run_training(SG, lw; run_config = cfg, output_config = out, test_mode = false)
        return L, PN, GN
    end

    function check_autodiff(name, make_lw)
        @testset "$name" begin
            L, PN, GN = autodiff_LPG(make_lw)
            @test all(isfinite, L)
            @test GN[end] > 0        # non-zero gradient ⇒ Enzyme propagated
            @test L[end] <= L[1]     # gradient step didn't increase loss (seed sign ok)
        end
    end

    AD_CONST            && check_autodiff("ConstLinearLW",     sg -> ConstLinearLW(sg))
    (AD_ABR  && DATA_OK) && check_autodiff("NeuralABRLW",       sg -> NeuralABRLW(sg, MLPConfig(n_hidden = 1, width = 8)))
    (AD_ABRG && DATA_OK) && check_autodiff("NeuralABRLWGlobal", sg -> NeuralABRLWGlobal(sg, MLPConfig(n_hidden = 1, width = 8)))

    # NeuralLinearLW AD is blocked by the same zscore data gap (forward throws before
    # AD can even start). Recorded as broken; flips to an Unexpected Pass once data exists.
    if AD_LINEAR
        @testset "NeuralLinearLW (blocked: zscore data gap)" begin
            @test_broken (DATA_OK &&
                (autodiff_LPG(sg -> NeuralLinearLW(sg, MLPConfig(n_hidden = 1, width = 8); standard_scaling = true)); true))
        end
    end
end
