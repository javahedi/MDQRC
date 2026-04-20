"""
Basis utilities for spin-1/2 systems.

We represent basis states as integers in [0, 2^N - 1],
where each bit corresponds to a spin.

Convention:
- site index is 1-based (Julia style)
- bit 0 -> spin up (+1)
- bit 1 -> spin down (-1)
"""

# Hilbert space dimension
hilbertdim(N::Int) = 1 << N

"""
Return the bit (0 or 1) at a given site.
"""
@inline function bitat(state::Integer, site::Int, N::Int)
    shift = N - site
    return (state >> shift) & 0x1
end

"""
Flip the spin at a given site.
"""
@inline function flipbit(state::Integer, site::Int, N::Int)
    shift = N - site
    return state ⊻ (1 << shift)
end

"""
Return σ^z eigenvalue at a site (+1 or -1).
"""
@inline function zvalue(state::Integer, site::Int, N::Int)
    return bitat(state, site, N) == 0 ? 1.0 : -1.0
end

"""
Construct a product state.

Example:
- all spins up (default)
- all spins down
"""
function product_state(N::Int; up::Bool=true)
    ψ = zeros(ComplexF64, hilbertdim(N))
    idx = up ? 1 : hilbertdim(N)
    ψ[idx] = 1.0 + 0im
    return ψ
end