using MDQRC
using Test
using LinearAlgebra

@testset "TFIM Hamiltonian" begin

    N = 4
    model = TFIMChain(N, 1.0, 0.5, 0.0)

    ψ = product_state(N)
    out = similar(ψ)

    apply_hamiltonian!(out, model, ψ, 0.1)

    @test length(out) == length(ψ)

    # sanity: result should not be zero
    @test norm(out) > 0

end