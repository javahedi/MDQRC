"""
Local Z observables.

These are the "sensors" in MD-QRC:
they extract classical signals from the quantum system.
"""

struct LocalZ <: AbstractObservable
    sites::Vector{Int}
end

"""
Expectation value <Z_i>.
"""
function local_z(ψ::AbstractVector{ComplexF64}, site::Int, N::Int)
    val = 0.0

    @inbounds for state in 0:(hilbertdim(N)-1)
        val += zvalue(state, site, N) * abs2(ψ[state + 1])
    end

    return val
end

"""
Measure multiple local Z observables.
"""
function measure(obs::LocalZ, ψ::AbstractVector{ComplexF64}, model::TFIMChain)
    return [local_z(ψ, s, model.N) for s in obs.sites]
end