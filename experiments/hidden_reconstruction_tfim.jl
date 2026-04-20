using MDQRC
using Statistics

println("=== Hidden Variable Reconstruction Experiment ===")

# ----------------------------
# System setup
# ----------------------------
N = 6
model = TFIMChain(N, 1.0, 0.7, 0.0)
ψ0 = product_state(N)

# Drive
hz(t) = 0.5 * sin(0.4 * t)

# ----------------------------
# Observables
# ----------------------------
sensor_obs = LocalZ([1, 2])   # accessible qubits
target_obs = LocalZ([6])      # hidden qubit

# ----------------------------
# Generate data (streaming!)
# ----------------------------
stream = evolve_observables(
    model,
    ψ0,
    LocalZ([1, 2, 6]);   # measure all once, then split
    dt=0.05,
    steps=400,
    hz=hz,
    method=KrylovExp(m=12),
)

data = stream.data

# Split features / targets
features = data[:, 1:2]   # Z1, Z2
targets  = data[:, 3:3]   # Z6

# ----------------------------
# Build supervised dataset
# ----------------------------
delay = 1
depth = 8
horizon = 6   # time reconstruction 

ds = build_supervised_dataset(
    features,
    targets;
    delay=delay,
    depth=depth,
    horizon=horizon,
)

# ----------------------------
# Train/test split
# ----------------------------
split = train_test_split(ds.X, ds.Y; train_fraction=0.7)

# ----------------------------
# Train readout
# ----------------------------
W = ridge_fit(split.Xtrain, split.Ytrain; λ=1e-6)

Ypred_train = ridge_predict(split.Xtrain, W)
Ypred_test  = ridge_predict(split.Xtest, W)

# ----------------------------
# Metrics
# ----------------------------
train_error = nmse(Ypred_train, split.Ytrain)
test_error  = nmse(Ypred_test, split.Ytest)

# ----------------------------
# Results
# ----------------------------
println("Sensors → Target: [Z1, Z2] → Z6")
println("Delay = $delay, Depth = $depth, Horizon = $horizon")
println("Train NMSE = $train_error")
println("Test  NMSE = $test_error")