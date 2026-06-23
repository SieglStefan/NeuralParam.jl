### Metrics + tree-norm utilities (pure, exact)

@testset "metrics" begin
    # rmse(x, y) = sqrt(mean((y - x)^2))
    @test rmse([1f0, 2f0], [1f0, 2f0]) == 0f0
    @test rmse([0f0, 0f0], [3f0, 4f0]) ≈ sqrt((9 + 16) / 2)

    # bias(x, y) = mean(y - x)
    @test bias([1f0, 2f0], [3f0, 5f0]) ≈ 2.5f0          # (2 + 3) / 2

    # correlation = cor(vec, vec)
    @test correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) ≈ 1.0   # perfectly linear

    # maxdiff(x, y) = max|y - x|
    @test maxdiff([1f0, 2f0], [1f0, 5f0]) == 3f0

    # tree norms over numbers / arrays / NamedTuples
    @test tree_l2sum(3.0)            ≈ 9.0
    @test tree_l2sum([3f0, 4f0])     ≈ 25f0
    @test tree_l2norm((; a = [3f0], b = [4f0])) ≈ 5f0          # 3-4-5
    @test tree_l2norm((; a = [3f0], b = [4f0])) ≈
          sqrt(tree_l2sum((; a = [3f0], b = [4f0])))
end
