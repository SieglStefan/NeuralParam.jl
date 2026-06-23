### RunConfig (hyperparameters) + OutputConfig (outputs).

@testset "config" begin

    @testset "RunConfig" begin
        c = RunConfig()
        @test c.eta0 == 1f-3
        @test c.n_ic == 10
        @test 0f0 < c.eta_decay <= 1f0
        @test c.patience >= 1

        # the :test preset shrinks everything to a 1-step smoke run
        ct = RunConfig(Val(:test))
        @test ct.n_ic     == 1
        @test ct.n_traj   == 1
        @test ct.n_epochs == 1
        @test ct.n_steps  == 1
        @test ct.n_gap    == 1
        @test ct.t_spinup == Day(1)
    end

    @testset "OutputConfig" begin
        o = OutputConfig()
        @test o.train_file == "training.csv"
        @test o.param_file == "param.jld2"
        @test o.save_path  === nothing

        # regression guard: save_path is Union{Nothing,String} (was a parse error).
        # Both `nothing` and a String must construct.
        @test OutputConfig(save_path = "out").save_path == "out"
    end
end
