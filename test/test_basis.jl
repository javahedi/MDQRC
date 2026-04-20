
using MDQRC
using Test

@testset "Basis utilities" begin

    N = 4

    @test hilbertdim(N) == 16

    # test bit access
    state = Int(0b0101)  # binary 0101
    @test bitat(state, 1, N) == 0
    @test bitat(state, 2, N) == 1

    # test flip
    flipped = flipbit(state, 2, N)
    @test flipped != state

    # test zvalue
    @test zvalue(0, 1, N) == 1.0   # all up
    @test zvalue(15, 1, N) == -1.0 # all down

    # product state
    ψ = product_state(N)
    @test length(ψ) == 16
    @test ψ[1] == 1.0 + 0im
    @test sum(abs2, ψ) ≈ 1.0

end