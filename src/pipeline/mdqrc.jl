"""
End-to-end MD-QRC pipeline.

This layer connects:
    quantum evolution -> observable stream -> delay embedding -> readout
"""

"""
Prepare supervised learning data from an observable time series.

Arguments:
- data: observable matrix of size (T, M)
- delay: embedding delay
- depth: embedding depth
- horizon: forecasting horizon
- target_cols: which columns to predict

Returns:
Named tuple:
- X
- Y
"""
function build_forecasting_dataset(
    data::AbstractMatrix;
    delay::Int,
    depth::Int,
    horizon::Int,
    target_cols=:,
)
    T, M = size(data)
    max_shift = (depth - 1) * delay
    nrows = T - max_shift - horizon

    nrows > 0 || error("Time series too short for given delay/depth/horizon.")

    target_idx = target_cols === Colon() ? collect(1:M) : collect(target_cols)
    ntarget = length(target_idx)

    X = Matrix{Float64}(undef, nrows, M * depth)
    Y = Matrix{Float64}(undef, nrows, ntarget)

    for r in 1:nrows
        t0 = r + max_shift

        col = 1
        for d in 0:(depth - 1)
            src_t = t0 - d * delay
            X[r, col:(col + M - 1)] .= data[src_t, :]
            col += M
        end

        Y[r, :] .= data[t0 + horizon, target_idx]
    end

    return (; X, Y)
end

"""
Split dataset into train/test parts.

Arguments:
- X, Y: supervised dataset
- train_fraction: fraction used for training

Returns:
Named tuple:
- Xtrain, Ytrain, Xtest, Ytest
"""
function train_test_split(
    X::AbstractMatrix,
    Y::AbstractMatrix;
    train_fraction::Float64=0.7,
)
    n = size(X, 1)
    ntrain = floor(Int, train_fraction * n)

    1 <= ntrain < n || error("Invalid train_fraction for dataset size.")

    return (
        Xtrain = X[1:ntrain, :],
        Ytrain = Y[1:ntrain, :],
        Xtest  = X[(ntrain + 1):end, :],
        Ytest  = Y[(ntrain + 1):end, :],
    )
end

"""
Mean squared error.
"""
mse(ŷ::AbstractMatrix, y::AbstractMatrix) = sum(abs2, ŷ .- y) / length(y)

"""
Normalized mean squared error.
"""
function nmse(ŷ::AbstractMatrix, y::AbstractMatrix)
    denom = sum(abs2, y) / length(y)
    denom > 0 || error("Cannot compute NMSE: target variance is zero.")
    return mse(ŷ, y) / denom
end

"""
Full MD-QRC forecasting pipeline.

Returns a named tuple containing:
- times
- data
- Xtrain, Ytrain, Xtest, Ytest
- W
- Ypred_train, Ypred_test
- train_nmse, test_nmse
"""
function mdqrc_forecast(
    model::TFIMChain,
    ψ0::AbstractVector{ComplexF64},
    observable::AbstractObservable;
    dt::Float64,
    steps::Int,
    hz::Function,
    delay::Int,
    depth::Int,
    horizon::Int,
    method::AbstractEvolutionMethod=RK4(),
    saveevery::Int=1,
    λ::Float64=1e-6,
    train_fraction::Float64=0.7,
    target_cols=:,
)
    stream = evolve_observables(
        model,
        ψ0,
        observable;
        dt=dt,
        steps=steps,
        hz=hz,
        method=method,
        saveevery=saveevery,
    )

    dataset = build_forecasting_dataset(
        stream.data;
        delay=delay,
        depth=depth,
        horizon=horizon,
        target_cols=target_cols,
    )

    split = train_test_split(dataset.X, dataset.Y; train_fraction=train_fraction)

    W = ridge_fit(split.Xtrain, split.Ytrain; λ=λ)

    Ypred_train = ridge_predict(split.Xtrain, W)
    Ypred_test  = ridge_predict(split.Xtest, W)

    return (
        times = stream.times,
        data = stream.data,
        Xtrain = split.Xtrain,
        Ytrain = split.Ytrain,
        Xtest = split.Xtest,
        Ytest = split.Ytest,
        W = W,
        Ypred_train = Ypred_train,
        Ypred_test = Ypred_test,
        train_nmse = nmse(Ypred_train, split.Ytrain),
        test_nmse = nmse(Ypred_test, split.Ytest),
    )
end