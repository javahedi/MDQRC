using MDQRC
using Test
using LinearAlgebra

@testset "RK4 Evolution" begin

    N = 4
    model = TFIMChain(N, 1.0, 0.5, 0.0)
    ψ0 = product_state(N)

    hz(t) = 0.2

    result = evolve(
        model,
        ψ0;
        dt=0.01,
        steps=10,
        hz=hz
    )

    @test length(result.times) == length(result.states)
    @test length(result.states) > 1

    # check normalization
    for ψ in result.states
        @test norm(ψ) ≈ 1.0 atol=1e-6
    end

end