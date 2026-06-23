### IO: csv logging round-trip + save/load round-trip.
### Qualify io calls with `NeuralParam.` (save/load names can collide with other deps).

@testset "io" begin
    tmp = mktempdir()

    @testset "csv round-trip" begin
        meta = Dict("scheme" => "NeuralABRLW", "n_in" => 20)
        NeuralParam.csv_init(meta; path = tmp, file = "t.csv")
        NeuralParam.csv_row!(1, 1, 1, 0.50f0, 1f-3, 2.0f0, 3.0f0; path = tmp, file = "t.csv")
        NeuralParam.csv_row!(1, 1, 2, 0.40f0, 1f-3, 2.1f0, 2.9f0; path = tmp, file = "t.csv")

        df = NeuralParam.csv_read(; path = tmp, file = "t.csv")
        # header must have NO leading spaces, else columns become " loss" and df.loss fails
        @test propertynames(df) == [:ic, :traj, :epoch, :loss, :eta, :pnorm, :gnorm]
        @test size(df, 1) == 2
        @test df.loss[2] ≈ 0.40f0

        info = NeuralParam.csv_info(; path = tmp, file = "t.csv")
        @test info["scheme"] == "NeuralABRLW"
        @test info["n_in"]   == "20"
    end

    @testset "save/load ConstLinearLW (value fidelity)" begin
        clw = ConstLinearLW(SG)
        clw = NeuralParam.update_ps(clw, (; a = collect(1f0:NLAYERS), b = collect(1f0:NLAYERS) .* 2f0))

        fp = NeuralParam.save(clw; path = tmp, file = "clw.jld2")
        @test isfile(fp)

        back = NeuralParam.load(; path = tmp, file = "clw.jld2")
        @test back isa ConstLinearLW
        @test back.ps.a == clw.ps.a
        @test back.ps.b == clw.ps.b
        @test back.scaling.sc_a == clw.scaling.sc_a
    end

    # Global scheme save/load exercises the to_cpu/cpu_device device path
    # (guards the bug where to_cpu(::NeuralABRLWGlobal) used undefined vars / wrong API).
    if DATA_OK
        @testset "save/load NeuralABRLWGlobal (device round-trip)" begin
            g  = NeuralABRLWGlobal(SG, MLPConfig(n_hidden = 1, width = 4))
            fp = NeuralParam.save(g; path = tmp, file = "g.jld2")
            @test isfile(fp)

            back = NeuralParam.load(; path = tmp, file = "g.jld2")
            @test back isa NeuralABRLWGlobal
            @test back.n_in     == g.n_in
            @test back.n_out    == g.n_out
            @test back.n_points == g.n_points
            @test back.ps == g.ps                      # values survived the round-trip
        end
    end
end
