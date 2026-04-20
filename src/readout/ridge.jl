using LinearAlgebra

"""
Ridge regression:

W = (XᵀX + λI)^(-1) XᵀY
"""
function ridge_fit(X::AbstractMatrix, Y::AbstractMatrix; λ::Float64=1e-6)
    d = size(X, 2)

    W = (X' * X + λ * I(d)) \ (X' * Y)
    return W
end

"""
Prediction
"""
function ridge_predict(X::AbstractMatrix, W::AbstractMatrix)
    return X * W
end