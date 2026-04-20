using MDQRC
using Test


@testset "Ridge Regression" begin

    X = rand(100, 10)
    Y = rand(100, 2)

    W = ridge_fit(X, Y)

    Y_pred = ridge_predict(X, W)

    @test size(Y_pred) == size(Y)

end