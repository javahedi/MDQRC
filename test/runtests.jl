# test/runtests.jl

using MDQRC
using Test

@testset "MDQRC.jl" begin
    include("test_basis.jl")
    include("test_tfim.jl")
    include("test_rk4.jl")
    include("test_krylov.jl")
    include("test_observables.jl")
    include("test_embedding.jl")
    include("test_readout.jl")
    include("test_pipeline.jl")
    include("test_supervised.jl")
end