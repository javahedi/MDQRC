"""
General supervised dataset builder for MD-QRC.

This allows mapping:
    feature time-series → target time-series

Core idea:
    X(t) = [x(t), x(t-τ), ..., x(t-(d-1)τ)]
    Y(t) = y(t + horizon)

This is more general than forecasting, and enables:
- hidden-variable reconstruction
- cross-observable prediction
- external target prediction
"""

function build_supervised_dataset(
    features::AbstractMatrix,
    targets::AbstractMatrix;
    delay::Int,
    depth::Int,
    horizon::Int,
)
    Tf, Mf = size(features)
    Tt, Mt = size(targets)

    Tf == Tt || error("Features and targets must have same time length.")

    max_shift = (depth - 1) * delay
    nrows = Tf - max_shift - horizon

    nrows > 0 || error("Time series too short for given parameters.")

    X = Matrix{Float64}(undef, nrows, Mf * depth)
    Y = Matrix{Float64}(undef, nrows, Mt)

    for r in 1:nrows
        t0 = r + max_shift

        # Build embedding from features
        col = 1
        for d in 0:(depth - 1)
            src_t = t0 - d * delay
            X[r, col:(col + Mf - 1)] .= features[src_t, :]
            col += Mf
        end

        # Target at future time
        Y[r, :] .= targets[t0 + horizon, :]
    end

    return (; X, Y)
end