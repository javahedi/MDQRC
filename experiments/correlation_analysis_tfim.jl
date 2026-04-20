using MDQRC
using Statistics
using Random
using DelimitedFiles

println("=== Correlation Analysis for Cross-Prediction ===")

# ----------------------------
# Config
# ----------------------------

rng = MersenneTwister(42)
N = 10
Jzz = 1.0 .* ones(N-1)
hz0 = -0.5 .+ 0.2 .* randn(rng, N)
model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])

ψ0 = product_state(N)

T = 3000
warmup = 100
ain = 0.01
dt = 0.05

sensor_sites = [1,2,3,4]
target_sites = [5,6,7,8,9,10]
lags = [0, 5, 10]

rng = MersenneTwister(42)

# ----------------------------
# Input
# ----------------------------
inputs_raw = randn(rng, T)
inputs = (inputs_raw .- mean(inputs_raw)) ./ std(inputs_raw)

function hz(t)
    k = Int(clamp(floor(t / dt) + 1, 1, length(inputs)))
    return ain * inputs[k]
end

# ----------------------------
# Measure all needed observables
# ----------------------------
obs = LocalZ(vcat(sensor_sites, target_sites))

stream = evolve_observables(
    model,
    ψ0,
    obs;
    dt=dt,
    steps=T-1,
    hz=hz,
    method=KrylovExp(m=20),
)

data = stream.data[(warmup + 1):end, :]

ns = length(sensor_sites)
nt = length(target_sites)

sensor_data = data[:, 1:ns]
target_data = data[:, (ns+1):(ns+nt)]

# ----------------------------
# Equal-time correlation matrix
# ----------------------------
corr0 = Matrix{Float64}(undef, ns, nt)

for i in 1:ns
    for j in 1:nt
        corr0[i, j] = cor(sensor_data[:, i], target_data[:, j])
    end
end

mkpath("results_correlation")
writedlm("results_correlation/corr_equal_time.csv", corr0, ',')

# ----------------------------
# Lagged correlations
# corr(sensor_i(t), target_j(t+lag))
# ----------------------------
for lag in lags
    corrmat = Matrix{Float64}(undef, ns, nt)

    for i in 1:ns
        for j in 1:nt
            x = sensor_data[1:(end-lag), i]
            y = target_data[(1+lag):end, j]
            corrmat[i, j] = cor(x, y)
        end
    end

    fname = "results_correlation/corr_lag_$(lag).csv"
    writedlm(fname, corrmat, ',')
end

println("Saved:")
println("  results_correlation/corr_equal_time.csv")
for lag in lags
    println("  results_correlation/corr_lag_$(lag).csv")
end