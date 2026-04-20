"""
Krylov-based time evolution for Schrödinger dynamics.

We approximate:

    ψ(t + dt) = exp(-im * H * dt) ψ(t)

using an Arnoldi Krylov subspace built from repeated applications of H to ψ,
without constructing the full Hamiltonian matrix.

Design goals:
- matrix-free
- low memory
- same public evolve(...) API style as RK4
- simple first implementation, extensible later

Notes:
- This implementation assumes the Hamiltonian is approximately constant over one
  time step dt, i.e. piecewise-constant driving is acceptable.
- The Krylov dimension m is fixed by the method object.
"""

struct KrylovExp <: AbstractEvolutionMethod
    m::Int
end

KrylovExp(; m::Int=20) = KrylovExp(m)

"""
Compute y = exp(-im * dt * H) * ψ using an Arnoldi Krylov approximation.

Arguments:
- model: quantum model
- ψ: current statevector
- hzval: scalar longitudinal field used during this step
- dt: time step
- m: Krylov dimension

Returns:
- propagated statevector (new allocation)
"""
function krylov_exp_step(
    model::TFIMChain,
    ψ::AbstractVector{ComplexF64},
    hzval::Float64,
    dt::Float64,
    m::Int,
)
    n = length(ψ)
    β = norm(ψ)

    β == 0 && error("Input state has zero norm.")

    # Trivial case
    if m < 1
        return copy(ψ)
    end

    # Arnoldi basis and Hessenberg matrix
    V = zeros(ComplexF64, n, m)
    Hm = zeros(ComplexF64, m, m)
    w = zeros(ComplexF64, n)

    V[:, 1] .= ψ ./ β

    k_eff = m

    for j in 1:m
        apply_hamiltonian!(w, model, view(V, :, j), hzval)

        # Modified Gram-Schmidt
        for i in 1:j
            hij = dot(view(V, :, i), w)
            Hm[i, j] = hij
            @. w = w - hij * V[:, i]
        end

        if j < m
            hnext = norm(w)

            # Breakdown: Krylov space closed early
            if hnext < 1e-14
                k_eff = j
                break
            end

            Hm[j + 1, j] = hnext
            V[:, j + 1] .= w ./ hnext
        end
    end

    # Truncate to the actually used Krylov dimension
    Veff = @view V[:, 1:k_eff]
    Heff = Hm[1:k_eff, 1:k_eff]

    # e1 vector in Krylov basis
    e1 = zeros(ComplexF64, k_eff)
    e1[1] = β

    # Exponential action in reduced space
    y_small = exp(-1im * dt * Heff) * e1
    ψnext = Veff * y_small

    # Normalize to control numerical drift
    ψnext ./= norm(ψnext)

    return ψnext
end

"""
User-facing evolution with Krylov exponential propagation.

Returns:
- times
- state trajectory
"""
function _evolve(
    method::KrylovExp,
    model::TFIMChain,
    ψ0::AbstractVector{ComplexF64};
    dt,
    steps,
    hz,
    saveevery=1,
)
    ψ = copy(ψ0)

    times = Float64[]
    states = Vector{Vector{ComplexF64}}()

    push!(times, 0.0)
    push!(states, copy(ψ))

    t = 0.0
    for step in 1:steps
        hzval = hz(t)
        ψ = krylov_exp_step(model, ψ, hzval, dt, method.m)
        t += dt

        if step % saveevery == 0
            push!(times, t)
            push!(states, copy(ψ))
        end
    end

    return (; times, states)
end