using MDQRC
using Test
using LinearAlgebra

@testset "Krylov Evolution" begin

    N = 4
    Jzz = 1.0 .* ones(N-1)
    hz0 = -0.5 .* ones(N)
    model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])

    ψ0 = product_state(N)

    hz(t) = 0.2

    result = evolve(
        model,
        ψ0;
        dt=0.01,
        steps=10,
        hz=hz,
        method=KrylovExp(m=8),
    )

    @test length(result.times) == length(result.states)
    @test length(result.states) > 1

    # Norm preservation
    for ψ in result.states
        @test norm(ψ) ≈ 1.0 atol=1e-8
    end
end


@testset "Krylov vs RK4 consistency" begin

    N = 4
    Jzz = 1.0 .* ones(N-1)
    hz0 = -0.5 .* ones(N)
    model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])
    ψ0 = product_state(N)

    hz(t) = 0.2
    dt = 1e-3
    steps = 5

    rk = evolve(
        model,
        ψ0;
        dt=dt,
        steps=steps,
        hz=hz,
        method=RK4(),
    )

    kr = evolve(
        model,
        ψ0;
        dt=dt,
        steps=steps,
        hz=hz,
        method=KrylovExp(m=10),
    )

    ψ_rk = rk.states[end]
    ψ_kr = kr.states[end]

    # Up to a global phase, compare overlap
    overlap = abs(dot(ψ_rk, ψ_kr))
    @test overlap ≈ 1.0 atol=1e-6
end