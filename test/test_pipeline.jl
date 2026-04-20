using MDQRC
using Test

@testset "Observable streaming" begin
    N=4
    Jzz = 1.0 .* ones(N-1)
    hz0 = -0.5 .* ones(N)
    model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])
    ψ0 = product_state(N)
    obs = LocalZ([1, 2])

    hz(t) = 0.2

    result = evolve_observables(
        model,
        ψ0,
        obs;
        dt=0.01,
        steps=20,
        hz=hz,
        method=RK4(),
        saveevery=2,
    )

    @test length(result.times) == size(result.data, 1)
    @test size(result.data, 2) == 2
end

@testset "Forecast dataset builder" begin
    data = rand(100, 3)

    ds = build_forecasting_dataset(
        data;
        delay=1,
        depth=5,
        horizon=1,
    )

    @test size(ds.X, 2) == 15
    @test size(ds.Y, 2) == 3
    @test size(ds.X, 1) == size(ds.Y, 1)
end

@testset "MD-QRC forecasting pipeline" begin
    N=4
    Jzz = 1.0 .* ones(N-1)
    hz0 = -0.5 .* ones(N)
    model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])
    ψ0 = product_state(N)
    obs = LocalZ([1, 2])

    hz(t) = 0.3 * sin(t)

    result = mdqrc_forecast(
        model,
        ψ0,
        obs;
        dt=0.02,
        steps=120,
        hz=hz,
        delay=1,
        depth=4,
        horizon=1,
        method=RK4(),
        λ=1e-6,
        train_fraction=0.7,
    )

    @test size(result.Ypred_test) == size(result.Ytest)
    @test result.train_nmse >= 0
    @test result.test_nmse >= 0
end