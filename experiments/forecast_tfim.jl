

using MDQRC
using Statistics
using Random

rng = MersenneTwister(42)
N = 10
Jzz = 1.0 .* ones(N-1)
hz0 = -0.5 .+ 0.2 .* randn(rng, N)
model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])

ψ0 = product_state(6)
obs = LocalZ([1, 2, 3])

hz(t) = 0.5 * sin(0.4 * t)

result = mdqrc_forecast(
    model,
    ψ0,
    obs;
    dt=0.05,
    steps=400,
    hz=hz,
    delay=1,
    depth=8,
    horizon=1,
    method=KrylovExp(m=12),
    λ=1e-6,
    train_fraction=0.7,
)

println("Train NMSE = ", result.train_nmse)
println("Test  NMSE = ", result.test_nmse)
println("Observable data shape = ", size(result.data))
println("Train set shape = ", size(result.Xtrain), " -> ", size(result.Ytrain))
println("Test  set shape = ", size(result.Xtest), " -> ", size(result.Ytest))