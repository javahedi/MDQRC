"""
Unified evolution interface.

Users call `evolve(...)` with a method.
Dispatch selects the correct backend (RK4, Krylov, ...).
"""

function evolve(
    model::TFIMChain,
    ψ0::AbstractVector{ComplexF64};
    dt::Float64,
    steps::Int,
    hz::Function,
    method::AbstractEvolutionMethod=RK4(),
    saveevery::Int=1,
)
    return _evolve(method, model, ψ0; dt=dt, steps=steps, hz=hz, saveevery=saveevery)
end

# fallback (important for debugging)
function _evolve(method::AbstractEvolutionMethod, args...; kwargs...)
    error("Evolution method $(typeof(method)) not implemented.")
end