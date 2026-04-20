# experimnts/memory_capacity_depth_sweep.jl

using MDQRC
using Statistics
using Plots
using Random

println("=== Memory Capacity Depth Sweep (MD-QRC) ===")

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
# Input signal
# ----------------------------
T = 3000
rng = MersenneTwister(42)
inputs = randn(rng, T)

# normalize (important)
inputs = (inputs .- mean(inputs)) ./ std(inputs)

dt = 0.05
ain = 0.01   # or even 0.001
function hz(t)
    k = Int(clamp(floor(t/dt) + 1, 1, length(inputs)))
    return ain * inputs[k]
end

# ----------------------------
# Observables
# ----------------------------
obs = LocalZ([1,2, 3, 4])

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
    method=KrylovExp(m=20),
)

features = stream.data

# ----------------------------
# Warmup removal
# ----------------------------
warmup = 100
features = features[warmup:end, :]
inputs = inputs[warmup:end]
T = length(inputs)

# ----------------------------
# Experiment parameters
# ----------------------------
delay = 1
dmax = 20
depth_list = [8, 12, 16, 20]

results = Dict()

# ----------------------------
# Main loop over depth
# ----------------------------
for depth in depth_list

    println("\n--- Depth = $depth ---")

    R2_vals = Float64[]

    for d in 0:dmax

        max_shift = (depth - 1) * delay

        start = max(max_shift + 1, d + 1)
        nrows = T - start + 1

        Mf = size(features, 2)

        X = Matrix{Float64}(undef, nrows, Mf * depth)
        Y = Vector{Float64}(undef, nrows)

        for r in 1:nrows
            t0 = start + r - 1

            # embedding
            col = 1
            for dd in 0:(depth-1)
                X[r, col:(col+Mf-1)] .= features[t0 - dd*delay, :]
                col += Mf
            end

            # target
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
    end

    C = sum(R2_vals)
    println("Total Capacity C = $C")

    results[depth] = (R2=R2_vals, C=C)
end

# ----------------------------
# Plot R²(d) curves
# ----------------------------
p1 = plot(
    xlabel="Delay d",
    ylabel="R²(d)",
    title="Memory Curve vs Depth",
    legend=:topright,
)

for depth in depth_list
    plot!(p1, 0:dmax, results[depth][:R2], marker=:o, label="depth=$depth")
end

savefig("R2.png")

# ----------------------------
# Plot Total Capacity
# ----------------------------
depths = collect(depth_list)
capacities = [results[d][:C] for d in depths]

p2 = plot(
    depths,
    capacities,
    marker=:o,
    xlabel="Embedding Depth",
    ylabel="Total Capacity CΣ",
    title="Capacity vs Depth",
    lw=2,
)

savefig("total_capacity.png")