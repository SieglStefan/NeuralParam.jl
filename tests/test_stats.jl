### Statistics: scalar z-score (+ broadcast), Scaling defaults, loading stats from disk.
### NOTE: zscore/inv_zscore are SCALAR now (the refactor removed the dots) — they are
### meant to be broadcast (`zscore.(x, μ, σ)`) in the parameterizations.

@testset "stats" begin

    @testset "zscore (scalar)" begin
        @test zscore(5f0, 2f0, 3f0)     ≈ 1f0          # (5 - 2) / 3
        @test inv_zscore(1f0, 2f0, 3f0) ≈ 5f0          # 1 * 3 + 2
        @test inv_zscore(zscore(7f0, 2f0, 3f0), 2f0, 3f0) ≈ 7f0   # roundtrip

        # broadcast form (as used in parameterization!)
        x = [1f0, 5f0, 9f0]
        @test inv_zscore.(zscore.(x, 2f0, 3f0), 2f0, 3f0) ≈ x
    end

    @testset "Scaling defaults" begin
        s = Scaling(NLAYERS)
        @test length(s.sc_a) == NLAYERS
        @test length(s.sc_b) == NLAYERS
        @test all(==(5f-8), s.sc_a)
        @test all(==(5f-6), s.sc_b)
    end

    # Loading stats from data/stats — pins the data contract. The ABR input length
    # (2*nlayers+4) vs NeuralLinearLW's input length (nlayers) is exactly why the
    # llw forward currently breaks while sharing the abrlw zscore file.
    if DATA_OK
        @testset "load ZScoreStats from disk" begin
            z = ZScoreStats("zscore_abrlw_L$(NLAYERS).jld2")
            @test length(z.input_mean)  == 2 * NLAYERS + 4
            @test length(z.input_std)   == 2 * NLAYERS + 4
            @test length(z.output_mean) == NLAYERS
            @test eltype(z.input_mean)  == Float32
            @test all(isfinite, z.input_std)
        end
    end
end
