"""
Delay embedding for MD-QRC.

Transforms a time-series of observables into a feature matrix:

X(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]

This is the "classical memory" of the system.
"""

"""
Build delay embedding.

Args:
- data: Matrix of size (T, M)
        T = time steps
        M = number of observables
- delay: τ
- depth: number of delays (d)

Returns:
- X: embedded feature matrix
"""
function delay_embedding(data::AbstractMatrix; delay::Int, depth::Int)
    T, M = size(data)

    max_shift = (depth - 1) * delay
    T_eff = T - max_shift

    T_eff <= 0 && error("Time series too short for embedding.")

    X = zeros(Float64, T_eff, M * depth)

    for t in 1:T_eff
        idx = 1
        for d in 0:(depth - 1)
            X[t, idx:(idx + M - 1)] .= data[t + d * delay, :]
            idx += M
        end
    end

    return X
end


"""
Construct prediction target.

Predict y(t + horizon)
"""
function build_target(data::AbstractMatrix; horizon::Int)
    T, M = size(data)

    T_eff = T - horizon
    T_eff <= 0 && error("Time series too short for target.")

    Y = data[(1 + horizon):(T), :]

    return Y
end