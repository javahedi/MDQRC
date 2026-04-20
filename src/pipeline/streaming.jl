"""
Streaming evolution for observables.

This avoids storing the full quantum state trajectory when the user only needs
the measured classical time series, which is the natural mode for MD-QRC.
"""

"""
Collect observable measurements during evolution.

Arguments:
- model: quantum model
- ψ0: initial state
- observable: measurement definition
- dt: timestep
- steps: number of evolution steps
- hz: driving function
- method: evolution backend
- saveevery: save/measure every `saveevery` steps

Returns:
Named tuple:
- times :: Vector{Float64}
- data  :: Matrix{Float64}

The returned matrix has size:
    (num_saved_times, num_observables)
"""
function evolve_observables(
    model::TFIMChain,
    ψ0::AbstractVector{ComplexF64},
    observable::AbstractObservable;
    dt::Float64,
    steps::Int,
    hz::Function,
    method::AbstractEvolutionMethod=RK4(),
    saveevery::Int=1,
)
    ψ = copy(ψ0)

    first_measurement = measure(observable, ψ, model)
    nobs = length(first_measurement)
    nsaved = fld(steps, saveevery) + 1

    times = Vector{Float64}(undef, nsaved)
    data = Matrix{Float64}(undef, nsaved, nobs)

    times[1] = 0.0
    data[1, :] .= first_measurement

    t = 0.0
    row = 2

    if method isa RK4
        ws = RK4Workspace(length(ψ))
        for step in 1:steps
            rk4_step!(ψ, model, t, dt, hz, ws)
            t += dt

            if step % saveevery == 0
                times[row] = t
                data[row, :] .= measure(observable, ψ, model)
                row += 1
            end
        end

    elseif method isa KrylovExp
        for step in 1:steps
            ψ = krylov_exp_step(model, ψ, hz(t), dt, method.m)
            t += dt

            if step % saveevery == 0
                times[row] = t
                data[row, :] .= measure(observable, ψ, model)
                row += 1
            end
        end

    else
        error("Observable streaming not implemented for method $(typeof(method)).")
    end

    return (; times, data)
end