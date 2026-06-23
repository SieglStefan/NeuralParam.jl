### General utilities

@testset "utils" begin
    # extract_layer(k, T) takes column k of each matrix in the series T
    T = [Float32[1 4; 2 5; 3 6], Float32[7 10; 8 11; 9 12]]

    @test extract_layer(1, T) == [Float32[1, 2, 3], Float32[7, 8, 9]]
    @test extract_layer(2, T) == [Float32[4, 5, 6], Float32[10, 11, 12]]
end
