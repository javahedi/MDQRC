
using MDQRC
using Test
using LinearAlgebra

@testset "Supervised dataset builder" begin

    features = rand(100, 2)
    targets  = rand(100, 1)

    ds = build_supervised_dataset(
        features,
        targets;
        delay=1,
        depth=5,
        horizon=2,
    )

    @test size(ds.X, 2) == 10
    @test size(ds.Y, 2) == 1
    @test size(ds.X, 1) == size(ds.Y, 1)

end