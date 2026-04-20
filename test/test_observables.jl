using MDQRC
using Test

@testset "Local Z observables" begin

    N = 4
    Jzz = 1.0 .* ones(N-1)
    hz0 = -0.5 .* ones(N)
    model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])
    ψ = product_state(N)

    obs = LocalZ([1, 2])

    vals = measure(obs, ψ, model)

    @test length(vals) == 2

    # all spins up → Z = +1
    @test vals[1] ≈ 1.0
    @test vals[2] ≈ 1.0

end