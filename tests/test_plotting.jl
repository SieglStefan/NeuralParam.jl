### Plotting smoke tests (opt-in: NEURALPARAM_TEST_PLOTTING=true).
### Heavy deps (Plots + CairoMakie + GeoMakie). We only assert the figures CONSTRUCT
### without erroring (no rendering / visual check). Heatmaps exercise the GeoMakie
### layout solver that used to StackOverflow.

@testset "plotting (smoke)" begin
    L  = Float32[1.0, 0.8, 0.6, 0.55]
    PN = Float32[2.0, 2.0, 2.1, 2.1]
    GN = Float32[1.0, 0.9, 0.8, 0.7]

    @testset "loss / training (vectors)" begin
        @test plot_loss(L)                                   !== nothing
        @test plot_training(L, PN, GN)                       !== nothing
        @test plot_training_comp([L, L .* 0.9f0]; labels = ["a", "b"]) !== nothing
    end

    @testset "loss / training (from csv)" begin
        tmp = mktempdir()
        NeuralParam.csv_init(Dict("scheme" => "X"); path = tmp, file = "t.csv")
        for (i, l) in enumerate(L)
            NeuralParam.csv_row!(1, 1, i, l, 1f-3, PN[i], GN[i]; path = tmp, file = "t.csv")
        end
        @test plot_loss(; path = tmp, file = "t.csv")     !== nothing
        @test plot_training(; path = tmp, file = "t.csv") !== nothing
    end

    @testset "heatmaps + comparison (real field)" begin
        model = PrimitiveWetModel(; spectral_grid = SG)
        sim   = initialize!(model)
        run!(sim, steps = 1)
        f = sim.variables.grid.temperature[:, SG.nlayers]

        @test plot_heatmap(f; coastlines = false)        !== nothing
        @test plot_heatmap(f; coastlines = true)         !== nothing
        @test plot_heatmaps([f, f]; coastlines = false)  !== nothing

        series = [f, f .+ 0.5f0, f .+ 0.3f0]
        @test plot_comparison((temperature = series,), (temperature = reverse(series),);
                              metric = rmse, Δt_sec = 3600) !== nothing
    end
end
