"""
Transverse-Field Ising Model (TFIM)

Hamiltonian:
    H(t) = J * ∑ Z_i Z_{i+1} + g * ∑ X_i + (hz0 + hz(t)) * ∑ Z_i

where:
- J   : nearest-neighbor ZZ interaction strength
- g   : transverse-field strength
- hz0 : static longitudinal field
- hz(t): externally supplied time-dependent drive / input signal

Design notes:
- `hz0` controls the intrinsic dynamical regime of the model
  (e.g. integrable vs nonintegrable behavior).
- `hz(t)` is the external input signal used by MD-QRC.
- The Hamiltonian is applied on-the-fly in a matrix-free manner.
"""
struct TFIMChain <: AbstractQuantumModel
    N::Int          # number of spins
    J::Float64      # ZZ interaction strength
    g::Float64      # transverse field strength
    hz0::Float64    # static longitudinal field
end

"""
Apply the TFIM Hamiltonian to a statevector in-place:
    out = H(t) * ψ

Arguments:
- `out`   : output buffer
- `model` : TFIMChain model parameters
- `ψ`     : input statevector
- `hz`    : external time-dependent longitudinal field at the current time step

Implementation:
- diagonal terms are computed directly from bit-string basis states
- off-diagonal X terms are generated via single-spin bit flips
- no dense Hamiltonian matrix is constructed
"""
function apply_hamiltonian!(
    out::AbstractVector{ComplexF64},
    model::TFIMChain,
    ψ::AbstractVector{ComplexF64},
    hz::Float64,
)
    N   = model.N
    J   = model.J
    g   = model.g
    hz0 = model.hz0

    fill!(out, 0)

    @inbounds for state in 0:(hilbertdim(N) - 1)
        amp = ψ[state + 1]
        amp == 0 && continue

        # -------------------------
        # Diagonal contribution
        # -------------------------
        diag = 0.0

        # Nearest-neighbor ZZ interaction
        for i in 1:(N - 1)
            diag += J * zvalue(state, i, N) * zvalue(state, i + 1, N)
        end

        # Static + time-dependent longitudinal field
        for i in 1:N
            diag += (hz0 + hz) * zvalue(state, i, N)
        end

        out[state + 1] += diag * amp

        # -------------------------
        # Off-diagonal contribution
        # -------------------------
        # Transverse X field flips one spin at a time
        for i in 1:N
            flipped = flipbit(state, i, N)
            out[flipped + 1] += g * amp
        end
    end

    return out
end