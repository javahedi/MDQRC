# experiments/nmse_vs_horizon_batch.jl


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

    function run_nmse_experiment(seed)
        rng = MersenneTwister(seed)
        
        Jzz = 1.0 .* ones(N-1)
        hz0 = -0.5 .+ 0.2 .* randn(rng, N)
        model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])
        ψ0 = product_state(N)

        inputs_raw = randn(rng, T)
        inputs = (inputs_raw .- mean(inputs_raw)) ./ std(inputs_raw)

        function hz(t)
            k = Int(clamp(floor(t / dt) + 1, 1, length(inputs)))
            return ain * inputs[k]
        end

        obs = LocalZ([1, 2, 3, 4])

        """
            That means:
                features = Z1, Z2, Z3, Z4  history
                target = future Z1
                
            Later you can make it harder by predicting:
            target_col = 4
            or a non-measured observable in a separate hidden-variable script
        """

        stream = evolve_observables(
            model,
            ψ0,
            obs;
            dt=dt,
            steps=T - 1,
            hz=hz,
            method=KrylovExp(m=20),
        )

        features_raw = stream.data

        features = features_raw[(warmup + 1):end, :]
        inputs_trimmed = inputs[(warmup + 1):end]

        Tlocal = length(inputs_trimmed)
        Mf = size(features, 2)

        max_shift = (depth - 1) * delay
        max_horizon = maximum(horizon_list)

        start_idx = max_shift + 1
        stop_idx = Tlocal - max_horizon
        nrows = stop_idx - start_idx + 1

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

        for τ in horizon_list
            # Target is the input at t0 + τ
            #Y = inputs_trimmed[(start_idx + τ):(stop_idx + τ)]  

            # Target is the future value of a measured observable
            target_col = 1 # predict future Z1
            Y = features[(start_idx + τ):(stop_idx + τ), target_col]

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
    mkpath("results_self_nmse/raw")

    println("Starting $n_runs runs on $(nprocs()) workers...")

    seeds = collect(1:n_runs)
    all_nmse = pmap(run_nmse_experiment, seeds)

    for (i, nmse_vals) in enumerate(all_nmse)
        filename = @sprintf("results_self_nmse/raw/run_%03d.csv", i)
        data = hcat(collect(horizon_list), nmse_vals)
        writedlm(filename, data, ',')
    end

    nmse_mean = [mean([run[i] for run in all_nmse]) for i in 1:length(horizon_list)]
    nmse_std  = [std([run[i] for run in all_nmse]) for i in 1:length(horizon_list)]

    summary_file = "results_self_nmse/self_nmse_summary.csv"
    open(summary_file, "w") do io
        println(io, "horizon,mean_nmse,std_nmse")
        for (i, τ) in enumerate(horizon_list)
            println(io, "$τ,$(nmse_mean[i]),$(nmse_std[i])")
        end
    end

    println("Experiment complete. Summary saved to $summary_file")
end

main()