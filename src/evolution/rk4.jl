"""
Fourth-order Runge-Kutta (RK4) evolution.

We solve:
    dψ/dt = -i H ψ

Design:
- Uses in-place operations
- Uses reusable workspace to avoid allocations
"""

struct RK4 <: AbstractEvolutionMethod end

"""
Workspace to avoid repeated allocations in RK4.
"""
struct RK4Workspace
    k1::Vector{ComplexF64}
    k2::Vector{ComplexF64}
    k3::Vector{ComplexF64}
    k4::Vector{ComplexF64}
    tmp::Vector{ComplexF64}
    htmp::Vector{ComplexF64}
end

function RK4Workspace(dim::Int)
    z = zeros(ComplexF64, dim)
    return RK4Workspace(copy(z), copy(z), copy(z), copy(z), copy(z), copy(z))
end

"""
Right-hand side of Schrödinger equation.
"""
function schrodinger_rhs!(
    out,
    ψ,
    model::TFIMChain,
    hz,
)
    apply_hamiltonian!(out, model, ψ, hz)
    @. out = -1im * out
end

"""
Single RK4 step (in-place).
"""
function rk4_step!(
    ψ,
    model::TFIMChain,
    t,
    dt,
    hz::Function,
    ws::RK4Workspace,
)
    k1 = ws.k1
    k2 = ws.k2
    k3 = ws.k3
    k4 = ws.k4
    tmp = ws.tmp
    htmp = ws.htmp

    h = hz(t)

    schrodinger_rhs!(k1, ψ, model, h)

    @. tmp = ψ + 0.5 * dt * k1
    schrodinger_rhs!(k2, tmp, model, h)

    @. tmp = ψ + 0.5 * dt * k2
    schrodinger_rhs!(k3, tmp, model, h)

    @. tmp = ψ + dt * k3
    schrodinger_rhs!(k4, tmp, model, h)

    @. ψ = ψ + (dt / 6) * (k1 + 2k2 + 2k3 + k4)

    # normalize to control drift
    ψ ./= norm(ψ)
end

"""
User-facing evolution function.

Returns:
- times
- state trajectory
"""
function _evolve(
    method::RK4,
    model::TFIMChain,
    ψ0::AbstractVector{ComplexF64};
    dt,
    steps,
    hz,
    saveevery=1,
)
    dim = length(ψ0)
    ws = RK4Workspace(dim)

    ψ = copy(ψ0)

    times = Float64[]
    states = Vector{Vector{ComplexF64}}()

    push!(times, 0.0)
    push!(states, copy(ψ))

    t = 0.0
    for step in 1:steps
        rk4_step!(ψ, model, t, dt, hz, ws)
        t += dt

        if step % saveevery == 0
            push!(times, t)
            push!(states, copy(ψ))
        end
    end

    return (; times, states)
end