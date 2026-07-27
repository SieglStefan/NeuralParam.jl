### Compile / smoke test for NeuralParam
###
### Run from the repo root:
###     julia --project=. tests/test_compile.jl
###
### Checks:
###   1. the package precompiles and loads
###   2. every EXPORTED name is actually defined (Julia allows exporting undefined names silently)
###   3. cheap runtime checks (construction + pure functions) — no simulation, no stats files

using NeuralParam
using Test
using SpeedyWeather: SpectralGrid

@testset "NeuralParam compile & smoke" begin

    @testset "all exported names are defined" begin
        exported = filter(!=(:NeuralParam), names(NeuralParam))
        undefined = [s for s in exported if !isdefined(NeuralParam, s)]
        isempty(undefined) || @warn "Exported but undefined" undefined
        @test isempty(undefined)
    end

    @testset "config construction" begin
        @test RunConfig() isa RunConfig
        @test RunConfig(Val(:test)) isa RunConfig     # exercises lw_scheme = nothing default
        @test OutputConfig() isa OutputConfig
    end

    @testset "metrics" begin
        x = Float32[1, 2, 3]; y = Float32[1, 2, 4]
        @test NeuralParam.mse(x, y)  ≈ 1/3
        @test NeuralParam.rmse(x, y) ≈ sqrt(1/3)
        @test NeuralParam.bias(x, y) ≈ 1/3
        @test NeuralParam.tree_l2norm((; a = Float32[3], b = 4f0)) ≈ 5
        # batch helpers used by the n_batch gradient averaging
        g1 = (; a = Float32[1, 2], b = 1f0)
        g2 = (; a = Float32[3, 4], b = 3f0)
        s  = NeuralParam.tree_add(g1, g2)
        @test s.a == Float32[4, 6] && s.b == 4f0
        h  = NeuralParam.tree_scale(s, 0.5f0)
        @test h.a == Float32[2, 3] && h.b == 2f0
    end

    @testset "stats math" begin
        @test NeuralParam.zscore(2f0, 1f0, 0.5f0) ≈ 2f0
        @test NeuralParam.inv_zscore(NeuralParam.zscore(3f0, 1f0, 2f0), 1f0, 2f0) ≈ 3f0
        sc = Scaling(8)
        @test length(sc.sc_a) == 8 && length(sc.sc_b) == 8
        @test NeuralParam.to_cpu(sc) isa Scaling          # device conversion still wired
    end

    @testset "utils" begin
        @test steps_from_days(1, 1800) == 48
        @test extract_layer(2, [Float32[1 2; 3 4]]) == [Float32[2, 4]]
    end

    @testset "paths" begin
        @test isdir(dirname(NeuralParam.model_dir("x")))     # results/models exists
        @test occursin("reference", NeuralParam.reference_dir("y"))
        @test occursin("stats",     NeuralParam.stats_dir("z"))
    end

    @testset "ConstLinearLW construct / update / info" begin
        sg = SpectralGrid(trunc = 31, nlayers = 8)
        s1 = ConstLinearLW(sg)
        @test s1 isa ConstLinearLW && length(s1.ps.a) == 8

        ps2 = (; a = fill(-2f0, 8), b = fill(0.5f0, 8))
        @test NeuralParam.update_ps(s1, ps2).ps.a == ps2.a

        info = NeuralParam.info_scheme(s1)
        @test info.scheme == "ConstLinearLW" && haskey(info, :init_a)
    end

    @testset "MLP architecture" begin
        cfg = MLPConfig()
        nn, ps, st = NeuralParam.setup_arch(cfg, 8, 16)
        @test nn !== nothing
        @test NeuralParam.info_arch(cfg).width == cfg.width
    end

    @testset "save / load roundtrip" begin
        sg  = SpectralGrid(trunc = 31, nlayers = 8)
        s   = ConstLinearLW(sg)
        dir = mktempdir()
        NeuralParam.save(s; path = dir, file = "s.jld2")
        s_loaded = NeuralParam.load(; path = dir, file = "s.jld2")
        @test s_loaded.ps.a == s.ps.a
    end

    @testset "write_info" begin
        dir = mktempdir()
        write_info(; path = dir, file = "i.toml", a = 1, b = 2.0, c = (; d = 3), e = [1, 2, 3])
        @test isfile(joinpath(dir, "i.toml"))
    end

    @testset "out dirs" begin
        base = mktempdir()
        d = prepare_out_dir(base, "run1")
        @test isdir(d)
        @test_throws ErrorException prepare_out_dir(base, "run1")   # overwrite protection
        @test isdir(fresh_out_dir(base, "run1"))                    # fresh_ overwrites
    end

end

println("COMPILE_OK")
