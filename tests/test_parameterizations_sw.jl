### Each parameterization must run inside a real SpeedyWeather simulation without
### producing NaN/Inf. This exercises parameterization!() end-to-end (indexing,
### input buffer, scaling, zscore) in the actual model — both the column schemes
### and the global (batched) scheme.

@testset "parameterizations in SpeedyWeather" begin

    # build a model with the scheme, run a few steps, return whether T stayed finite
    function run_finite(lw)
        model = PrimitiveWetModel(; spectral_grid = SG, longwave_radiation = lw)
        sim   = initialize!(model)
        run!(sim, steps = 2)
        return all(isfinite, Array(parent(sim.variables.grid.temperature)))
    end

    @testset "ConstLinearLW" begin
        @test run_finite(ConstLinearLW(SG))
    end

    if DATA_OK
        mlp = MLPConfig(n_hidden = 1, width = 8)

        @testset "NeuralABRLW (column)" begin
            # full ABR input set: T, q, pressure, SST, soil T, land mask
            @test run_finite(NeuralABRLW(SG, mlp))
        end

        @testset "NeuralABRLWGlobal (global/batched)" begin
            @test run_finite(NeuralABRLWGlobal(SG, mlp))
        end

        # KNOWN DATA GAP: NeuralLinearLW loads the length-(2n+4) abrlw zscore for its
        # length-n input → DimensionMismatch in the forward pass. Marked broken until a
        # proper llw zscore (input-only / with output) exists. Un-break when it passes.
        @testset "NeuralLinearLW (blocked: zscore data gap)" begin
            @test_broken run_finite(NeuralLinearLW(SG, mlp; standard_scaling = true))
        end
    end
end
