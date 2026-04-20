"""
Disordered Transverse-Field Ising Model (TFIM)

Hamiltonian:
    H(t) = ∑_{i=1}^{N-1} Jzz[i] * Z_i Z_{i+1}
         + g * ∑_{i=1}^{N} X_i
         + ∑_{i=1}^{N} hz0[i] * Z_i
         + hz(t) * ∑_{i ∈ drive_sites} Z_i

where:
- Jzz[i]       : nearest-neighbor ZZ couplings on bond (i, i+1)
- g            : uniform transverse-field strength
- hz0[i]       : static site-dependent longitudinal fields
- hz(t)        : time-dependent input signal
- drive_sites  : subset of sites where the input is applied

Design:
- site/bond disorder breaks mirror symmetry
- local driving allows directional information injection
- Hamiltonian is applied on-the-fly, no dense matrix construction
"""
struct TFIMChain <: AbstractQuantumModel
    N::Int
    Jzz::Vector{Float64}       # length N-1
    g::Float64
    hz0::Vector{Float64}       # length N
    drive_sites::Vector{Int}
end

"""
Convenience constructor for uniform couplings/fields.

Example:
    model = TFIMChain(10, 1.0, 1.05, -0.5, [1,2])

This creates:
- Jzz = fill(1.0, N-1)
- hz0 = fill(-0.5, N)
"""
function TFIMChain(
    N::Int,
    J::Real,
    g::Real,
    hz0::Real,
    drive_sites::Vector{Int},
)
    return TFIMChain(
        N,
        fill(Float64(J), N - 1),
        Float64(g),
        fill(Float64(hz0), N),
        drive_sites,
    )
end

"""
Validation helper for TFIMChain parameters.
"""
function validate(model::TFIMChain)
    length(model.Jzz) == model.N - 1 || error("Jzz must have length N-1.")
    length(model.hz0) == model.N     || error("hz0 must have length N.")
    all(1 .<= model.drive_sites .<= model.N) || error("drive_sites must be valid site indices.")
    return nothing
end

"""
Apply the disordered TFIM Hamiltonian to a statevector in-place:
    out = H(t) * ψ

Arguments:
- `out`   : output buffer
- `model` : TFIMChain model parameters
- `ψ`     : input statevector
- `hz`    : external time-dependent longitudinal drive at current time

Implementation:
- diagonal ZZ and Z terms are computed directly from the bit-string basis
- off-diagonal X terms are generated via single-spin bit flips
- no dense Hamiltonian matrix is constructed
"""
function apply_hamiltonian!(
    out::AbstractVector{ComplexF64},
    model::TFIMChain,
    ψ::AbstractVector{ComplexF64},
    hz::Float64,
)
    validate(model)

    N = model.N
    Jzz = model.Jzz
    g = model.g
    hz0 = model.hz0
    drive_sites = model.drive_sites

    fill!(out, 0)

    @inbounds for state in 0:(hilbertdim(N) - 1)
        amp = ψ[state + 1]
        amp == 0 && continue

        # -------------------------
        # Diagonal contribution
        # -------------------------
        diag = 0.0

        # Bond-disordered ZZ interaction
        for i in 1:(N - 1)
            diag += Jzz[i] * zvalue(state, i, N) * zvalue(state, i + 1, N)
        end

        # Site-disordered static longitudinal field
        for i in 1:N
            diag += hz0[i] * zvalue(state, i, N)
        end

        # Local time-dependent drive
        for i in drive_sites
            diag += hz * zvalue(state, i, N)
        end

        out[state + 1] += diag * amp

        # -------------------------
        # Off-diagonal contribution
        # -------------------------
        for i in 1:N
            flipped = flipbit(state, i, N)
            out[flipped + 1] += g * amp
        end
    end

    return out
end