
using MDQRC
using Statistics

println("=== Memory Capacity Experiment (MD-QRC) ===")

# ----------------------------
# System setup
# ----------------------------
rng = MersenneTwister(42)
N = 10
Jzz = 1.0 .* ones(N-1)
hz0 = -0.5 .+ 0.2 .* randn(rng, N)
model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])

ψ0 = product_state(N)

# ----------------------------
# Input signal (IMPORTANT)
# ----------------------------
T = 600
inputs = rand(T)   # uniform [0,1]

# map time index → continuous time
dt = 0.05
ain = 0.01   # or even 0.001
function hz(t)
    k = Int(clamp(floor(t/dt) + 1, 1, length(inputs)))
    return ain * inputs[k]
end

# ----------------------------
# Observables (sensors)
# ----------------------------
obs = LocalZ([1,2])

# ----------------------------
# Generate observable stream
# ----------------------------
stream = evolve_observables(
    model,
    ψ0,
    obs;
    dt=dt,
    steps=T-1,
    hz=hz,
    method=KrylovExp(m=12),
)

features = stream.data   # size (T, num_sensors)

# ----------------------------
# Parameters
# ----------------------------
delay = 1
depth = 8
dmax = 20

R2_vals = Float64[]

# ----------------------------
# Loop over memory delay
# ----------------------------
for d in 0:dmax

    # Build dataset manually
    max_shift = (depth - 1) * delay

    start = max(max_shift + 1, d + 1)
    nrows = T - start + 1

    X = Matrix{Float64}(undef, nrows, size(features,2)*depth)
    Y = Vector{Float64}(undef, nrows)

    Mf = size(features, 2)

    for r in 1:nrows
        t0 = start + r - 1

        # embedding
        col = 1
        for dd in 0:(depth-1)
            X[r, col:(col+Mf-1)] .= features[t0 - dd*delay, :]
            col += Mf
        end

        # target (SAFE now)
        Y[r] = inputs[t0 - d]
    end
 

    # split
    split = train_test_split(X, reshape(Y, :, 1); train_fraction=0.7)

    # train
    W = ridge_fit(split.Xtrain, split.Ytrain; λ=1e-6)

    Ypred = ridge_predict(split.Xtest, W)

    # compute R²
    y_true = vec(split.Ytest)
    y_pred = vec(Ypred)

    num = cov(y_true, y_pred)^2
    den = var(y_true) * var(y_pred)
    R2 = den > 0 ? num / den : 0.0

    push!(R2_vals, R2)

    println("d = $d → R² = $R2")
end

# ----------------------------
# Total capacity
# ----------------------------
C = sum(R2_vals)

println("\nTotal Memory Capacity = $C")