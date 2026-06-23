### MLP architecture config + setup_arch (internal).

@testset "models (MLP)" begin
    # defaults — `act` default is leakyrelu (guards that it stays in scope)
    c = MLPConfig()
    @test c.n_hidden == 2
    @test c.width    == 16
    @test c.act      === Lux.leakyrelu

    # overrides
    c2 = MLPConfig(n_hidden = 3, width = 8)
    @test c2.n_hidden == 3
    @test c2.width    == 8

    # setup_arch builds a usable Lux net of the requested in/out size
    n_in, n_out = 5, 3
    nn, ps, st = NeuralParam.setup_arch(MLPConfig(n_hidden = 1, width = 4), n_in, n_out, Random.Xoshiro(0))

    x = randn(Float32, n_in)
    y, _ = Lux.apply(nn, x, ps, st)
    @test length(y) == n_out
    @test eltype(y) == Float32
end
