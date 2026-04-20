
using MDQRC
using Test

@testset "Delay Embedding" begin

    data = rand(100, 3)

    X = delay_embedding(data; delay=1, depth=5)

    @test size(X, 2) == 15
    @test size(X, 1) == 96

end