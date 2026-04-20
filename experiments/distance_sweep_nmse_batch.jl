# experiments/distance_sweep_nmse_batch.jl

using Distributed

if nprocs() == 1
    addprocs()
end

@everywhere begin
    using MDQRC
    using Statistics
    using Random

    const N = 10
    const sensors = [1,2,3,4]
    const targets = [5,6,7,8,9,10]

    const depth = 12
    const delay = 1
    const τ = 10   # fixed prediction horizon

    const T = 3000
    const warmup = 100
    const ain = 0.01
    const dt = 0.05

    function run_distance_experiment(seed)
        rng = MersenneTwister(seed)      

        Jzz = 1.0 .* ones(N-1)
        hz0 = -0.5 .+ 0.2 .* randn(rng, N)
        model = TFIMChain(N, Jzz, 1.05, hz0, [1,2])
        
        ψ0 = product_state(N)

        # input
        inputs_raw = randn(rng, T)
        inputs = (inputs_raw .- mean(inputs_raw)) ./ std(inputs_raw)

        function hz(t)
            k = Int(clamp(floor(t/dt) + 1, 1, length(inputs)))
            return ain * inputs[k]
        end

        # measure sensors + all targets
        obs = LocalZ(vcat(sensors, targets))

        stream = evolve_observables(
            model, ψ0, obs;
            dt=dt,
            steps=T-1,
            hz=hz,
            method=KrylovExp(m=20),
        )

        data = stream.data[(warmup+1):end, :]
        Tlocal = size(data,1)

        # split features / targets
        features = data[:, 1:length(sensors)]
        Mf = size(features,2)

        results = Dict{Int, Float64}()

        for (idx, target_site) in enumerate(targets)

            target_col = length(sensors) + idx
            target = data[:, target_col]

            max_shift = (depth-1)*delay
            start_idx = max_shift + 1
            stop_idx = Tlocal - τ
            nrows = stop_idx - start_idx + 1

            X = Matrix{Float64}(undef, nrows, Mf*depth)
            Y = Vector{Float64}(undef, nrows)

            for r in 1:nrows
                t0 = start_idx + r - 1

                # embedding
                col = 1
                for dd in 0:(depth-1)
                    X[r, col:(col+Mf-1)] .= features[t0 - dd*delay, :]
                    col += Mf
                end

                Y[r] = target[t0 + τ]
            end

            split = train_test_split(X, reshape(Y, :, 1); train_fraction=0.7)

            W = ridge_fit(split.Xtrain, split.Ytrain; λ=1e-6)
            Ypred = ridge_predict(split.Xtest, W)

            mse = mean((Ypred .- split.Ytest).^2)
            power_y = mean(split.Ytest.^2)

            results[target_site] = mse / power_y
        end

        return results
    end
end

using DelimitedFiles
using Printf
using Statistics

function main()
    n_runs = 20
    mkpath("results_distance")

    seeds = collect(1:n_runs)
    all_results = pmap(run_distance_experiment, seeds)

    # aggregate
    summary_file = "results_distance/distance_nmse.csv"

    open(summary_file, "w") do io
        println(io, "site,mean_nmse,std_nmse")

        for site in targets
            vals = [run[site] for run in all_results]
            println(io, "$site,$(mean(vals)),$(std(vals))")
        end
    end

    println("Distance sweep complete → saved to $summary_file")
end

main()