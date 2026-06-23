### Parameterization construction + dimension arithmetic + update_ps.
### (No SpeedyWeather run here — that's test_parameterizations_sw.jl.)
### update_ps is the key guard: it must rebuild the scheme with the NEW ps while
### preserving nn/st/arch_config/zscore/scaling (this is where the nn_config→arch_config
### bug lived).

@testset "parameterizations (construction)" begin

    # ---- ConstLinearLW (needs no data file) ----
    @testset "ConstLinearLW" begin
        clw = ConstLinearLW(SG)
        @test clw isa SpeedyWeather.AbstractLongwave
        @test clw.ps.a == -ones(Float32, NLAYERS)      # baseline a = -1
        @test clw.ps.b ==  ones(Float32, NLAYERS)      # baseline b =  1
        @test length(clw.scaling.sc_a) == NLAYERS

        # update_ps: new object with swapped ps, scaling preserved, original untouched
        new_ps = (; a = ones(Float32, NLAYERS), b = 2 .* ones(Float32, NLAYERS))
        clw2 = NeuralParam.update_ps(clw, new_ps)
        @test clw2.ps.a == ones(Float32, NLAYERS)
        @test clw2.scaling === clw.scaling
        @test clw.ps.a == -ones(Float32, NLAYERS)      # original unchanged (immutability)
    end

    # ---- neural schemes (need the zscore fixture) ----
    if DATA_OK
        mlp = MLPConfig(n_hidden = 1, width = 8)

        @testset "NeuralLinearLW" begin
            nlw = NeuralLinearLW(SG, mlp; standard_scaling = true)
            @test nlw.n_in  == NLAYERS                 # temperature profile
            @test nlw.n_out == 2 * NLAYERS             # a and b per layer
            @test length(nlw.input_buffer) == nlw.n_in

            nlw2 = NeuralParam.update_ps(nlw, nlw.ps)
            @test nlw2.nn          === nlw.nn
            @test nlw2.st          === nlw.st
            @test nlw2.arch_config === nlw.arch_config
            @test nlw2.zscore      === nlw.zscore
            @test nlw2.scaling     === nlw.scaling
        end

        @testset "NeuralABRLW" begin
            abr = NeuralABRLW(SG, mlp)
            @test abr.n_in  == 2 * NLAYERS + 4         # T, q profiles + ps, SST, soil T, land
            @test abr.n_out == NLAYERS
            @test length(abr.input_buffer) == abr.n_in

            abr2 = NeuralParam.update_ps(abr, abr.ps)
            @test abr2.nn          === abr.nn
            @test abr2.arch_config === abr.arch_config
            @test abr2.zscore      === abr.zscore
        end

        @testset "NeuralABRLWGlobal" begin
            g = NeuralABRLWGlobal(SG, mlp)
            @test g.n_in     == 2 * NLAYERS + 4
            @test g.n_out    == NLAYERS
            @test g.n_points == SG.npoints

            g2 = NeuralParam.update_ps(g, g.ps)
            @test g2.nn          === g.nn
            @test g2.arch_config === g.arch_config
            @test g2.n_points    == g.n_points
        end
    end
end
