using MDQRC
using Test
using LinearAlgebra

@testset "TFIM Hamiltonian" begin

    N = 4
    Jzz = 1.0 .* ones(N-1)
    hz0 = -0.5 .* ones(N)
    model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])

    ψ = product_state(N)
    out = similar(ψ)

    apply_hamiltonian!(out, model, ψ, 0.1)

    @test length(out) == length(ψ)

    # sanity: result should not be zero
    @test norm(out) > 0

end