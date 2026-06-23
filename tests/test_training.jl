### Online training loop — PLUMBING ONLY (test_mode=true skips Enzyme).
### Real autodiff lives in test_autodiff.jl (slow, opt-in). This file stays fast:
### it guards the run_training signature/return contract and the csv logging path.

@testset "training (test_mode)" begin

    @testset "ConstLinearLW loop + csv" begin
        tmp = mktempdir()
        out = OutputConfig(
            printing_ic   = false,
            printing_traj = false,
            save_path     = tmp,
            param_save    = false,
            plotting      = false,
        )

        param, L, PN, GN = run_training(
            SG, ConstLinearLW(SG);
            run_config    = RunConfig(Val(:test)),
            output_config = out,
            test_mode     = true,
        )

        @test param isa ConstLinearLW
        @test L isa Vector{Float32}
        @test !isempty(L)
        @test all(isfinite, L)
        @test length(PN) == length(L)
        @test length(GN) == length(L)

        # the loop wrote the csv and it is re-readable with the right columns
        df = NeuralParam.csv_read(; path = tmp, file = out.train_file)
        @test :loss in propertynames(df)
        @test size(df, 1) == length(L)
    end

    # one neural scheme through the loop in test_mode: exercises the neural update_ps
    # rebuild + build_meta + the @set radiation swap.
    if DATA_OK
        @testset "NeuralABRLW loop (neural update_ps path)" begin
            out = OutputConfig(
                printing_ic   = false,
                printing_traj = false,
                save_path     = mktempdir(),
                train_save    = false,
                param_save    = false,
                plotting      = false,
            )

            param, L, PN, GN = run_training(
                SG, NeuralABRLW(SG, MLPConfig(n_hidden = 1, width = 8));
                run_config    = RunConfig(Val(:test)),
                output_config = out,
                test_mode     = true,
            )

            @test param isa NeuralABRLW
            @test all(isfinite, L)
        end
    end
end
