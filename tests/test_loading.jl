### Smoke test: does NeuralParam precompile, load, and do its cheap src work?
###
### Run from the repo root:
###     julia --project=. tests/test_loading.jl
###
### Only cheap checks here (construction + pure functions) — NO full SpeedyWeather
### simulation, NO stats-file loading. Meant to catch typos / signature / wiring bugs fast.

using NeuralParam
using Test
using SpeedyWeather: SpectralGrid

@testset "NeuralParam smoke tests" begin

    @testset "exports defined" begin
        for sym in (
            :ConstLinearLW, :NeuralLinearLW, :NeuralABRLW, :NeuralABRLWGlobal,
            :MLPConfig,
            :RunConfig, :OutputConfig, :run_training, :compute_metrics,
            :evaluate_rollout, :evaluate_benchmark,
            :save_scheme, :load_scheme, :save_reference, :load_reference,
            :write_info, :info_scheme,
            :zscore, :inv_zscore, :Scaling, :ZScoreStats,
            :mse, :rmse, :bias, :steps_from_days,
            :perturb_grid_field!, :sim_timesteps!, :sample_trajectory,
        )
            @test isdefined(NeuralParam, sym)
        end
    end

    @testset "config construction" begin
        @test RunConfig() isa RunConfig
        @test RunConfig(Val(:test)) isa RunConfig       # exercises lw_scheme = nothing default
        @test OutputConfig() isa OutputConfig
    end

    @testset "metrics" begin
        x = Float32[1, 2, 3]; y = Float32[1, 2, 4]
        @test mse(x, y)  ≈ 1/3
        @test rmse(x, y) ≈ sqrt(1/3)
        @test bias(x, y) ≈ 1/3
        @test maxdiff(x, y) == 1
        @test tree_l2norm((; a = Float32[3], b = 4f0)) ≈ 5
    end

    @testset "stats math" begin
        @test zscore(2f0, 1f0, 0.5f0) ≈ 2f0
        @test inv_zscore(zscore(3f0, 1f0, 2f0), 1f0, 2f0) ≈ 3f0
        sc = Scaling(8)
        @test length(sc.sc_a) == 8 && length(sc.sc_b) == 8
    end

    @testset "utils" begin
        @test steps_from_days(1, 1800) == 48
        @test extract_layer(2, [Float32[1 2; 3 4]]) == [Float32[2, 4]]
    end

    @testset "ConstLinearLW construct / update / info" begin
        sg = SpectralGrid(trunc = 31, nlayers = 8)
        s1 = ConstLinearLW(sg)
        @test s1 isa ConstLinearLW
        @test length(s1.ps.a) == 8

        ps2 = (; a = fill(-2f0, 8), b = fill(0.5f0, 8))
        s2 = NeuralParam.update_ps(s1, ps2)
        @test s2.ps.a == ps2.a

        info = info_scheme(s1)
        @test info.scheme == "ConstLinearLW"
        @test haskey(info, :init_a)
    end

    @testset "MLP architecture" begin
        cfg = MLPConfig()
        nn, ps, st = NeuralParam.setup_arch(cfg, 8, 16)
        @test nn !== nothing
        ia = NeuralParam.info_arch(cfg)
        @test ia.width == cfg.width
    end

    @testset "scheme IO roundtrip" begin
        sg = SpectralGrid(trunc = 31, nlayers = 8)
        s = ConstLinearLW(sg)
        dir = mktempdir()
        save_scheme(s; path = dir, file = "s.jld2")
        s_loaded = load_scheme(path = dir, file = "s.jld2")
        @test s_loaded.ps.a == s.ps.a
    end

    @testset "write_info" begin
        dir = mktempdir()
        write_info(; path = dir, file = "i.toml", a = 1, b = 2.0, c = (; d = 3), e = [1, 2, 3])
        @test isfile(joinpath(dir, "i.toml"))
    end

end

println("SMOKE_OK")
