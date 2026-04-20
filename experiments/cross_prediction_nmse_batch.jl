# experiments/cross_prediction_nmse_batch.jl


using Distributed

if nprocs() == 1
    addprocs()
end

@everywhere begin
    using MDQRC
    using Statistics
    using Random

    const N = 10
    const depth = 12
    const delay = 1
    const horizon_list = 0:20

    const T = 3000
    const warmup = 100
    const ain = 0.01
    const dt = 0.05

    """
    Run one cross-prediction experiment:

        [Z1,Z2,Z3,Z4]_history -> Z10(t+τ)

    Returns:
    - nmse_vals :: Vector{Float64}
    """
    function run_cross_prediction_experiment(seed)
        rng = MersenneTwister(seed)

        # ----------------------------
        # Physics setup
        # ----------------------------
        Jzz = 1.0 .* ones(N-1)
        hz0 = -0.5 .+ 0.2 .* randn(rng, N)
        model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])
        ψ0 = product_state(N)

        # ----------------------------
        # Input signal
        # ----------------------------
        inputs_raw = randn(rng, T)
        inputs = (inputs_raw .- mean(inputs_raw)) ./ std(inputs_raw)

        function hz(t)
            k = Int(clamp(floor(t / dt) + 1, 1, length(inputs)))
            return ain * inputs[k]
        end

        # ----------------------------
        # Measure sensors + target together
        # columns 1:4 = sensors, column 5 = target Z10
        # ----------------------------
        obs = LocalZ([1, 2, 3, 4, 10])

        stream = evolve_observables(
            model,
            ψ0,
            obs;
            dt=dt,
            steps=T - 1,
            hz=hz,
            method=KrylovExp(m=20),
        )

        data_raw = stream.data

        # ----------------------------
        # Warmup removal
        # ----------------------------
        data = data_raw[(warmup + 1):end, :]
        Tlocal = size(data, 1)

        features = data[:, 1:4]   # Z1..Z4
        target   = data[:, 5]     # Z10

        Mf = size(features, 2)

        max_shift = (depth - 1) * delay
        max_horizon = maximum(horizon_list)

        start_idx = max_shift + 1
        stop_idx = Tlocal - max_horizon
        nrows = stop_idx - start_idx + 1

        # ----------------------------
        # Precompute embedded feature matrix once
        # ----------------------------
        X_embedded = Matrix{Float64}(undef, nrows, Mf * depth)

        for r in 1:nrows
            t0 = start_idx + r - 1
            col = 1
            for dd in 0:(depth - 1)
                X_embedded[r, col:(col + Mf - 1)] .= features[t0 - dd * delay, :]
                col += Mf
            end
        end

        nmse_vals = Float64[]

        # ----------------------------
        # Horizon sweep
        # ----------------------------
        for τ in horizon_list
            Y = target[(start_idx + τ):(stop_idx + τ)]

            split = train_test_split(
                X_embedded,
                reshape(Y, :, 1);
                train_fraction=0.7,
            )

            W = ridge_fit(split.Xtrain, split.Ytrain; λ=1e-6)
            Ypred = ridge_predict(split.Xtest, W)

            mse = mean((Ypred .- split.Ytest) .^ 2)
            power_y = mean(split.Ytest .^ 2)

            push!(nmse_vals, mse / power_y)
        end

        return nmse_vals
    end
end

using DelimitedFiles
using Printf
using Statistics

function main()
    n_runs = 20
    outdir = "results_cross_nmse"
    rawdir = joinpath(outdir, "raw")
    mkpath(rawdir)

    println("Starting $n_runs cross-prediction runs on $(nprocs()) workers...")

    seeds = collect(1:n_runs)
    all_nmse = pmap(run_cross_prediction_experiment, seeds)

    # Save raw runs
    for (i, nmse_vals) in enumerate(all_nmse)
        filename = @sprintf("%s/run_%03d.csv", rawdir, i)
        data = hcat(collect(horizon_list), nmse_vals)
        writedlm(filename, data, ',')
    end

    # Summary
    nmse_mean = [mean([run[i] for run in all_nmse]) for i in 1:length(horizon_list)]
    nmse_std  = [std([run[i] for run in all_nmse]) for i in 1:length(horizon_list)]

    summary_file = joinpath(outdir, "cross_nmse_summary.csv")
    open(summary_file, "w") do io
        println(io, "horizon,mean_nmse,std_nmse")
        for (i, τ) in enumerate(horizon_list)
            println(io, "$τ,$(nmse_mean[i]),$(nmse_std[i])")
        end
    end

    println("Cross-prediction experiment complete.")
    println("Summary saved to $summary_file")
end

main()