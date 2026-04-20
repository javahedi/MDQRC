module MDQRC

    using LinearAlgebra

    include("types.jl")
    include("basis.jl")
    include("models/tfim.jl")
    include("evolution/api.jl") 
    include("evolution/rk4.jl")
    include("evolution/krylov.jl")
    include("observables/localz.jl")

    include("reservoir/embedding.jl")
    include("readout/ridge.jl")

    include("pipeline/streaming.jl")
    include("pipeline/mdqrc.jl")
    include("pipeline/supervised.jl")




    # exports (clean API)
    export AbstractQuantumModel
    export AbstractEvolutionMethod
    export AbstractObservable

    export hilbertdim, bitat, flipbit, zvalue
    export TFIMChain, apply_hamiltonian!
    export RK4, KrylovExp

    export product_state
    export evolve

    export LocalZ
    export measure

    export delay_embedding
    export build_target
    export ridge_fit
    export ridge_predict

    export evolve_observables
    export build_forecasting_dataset
    export train_test_split
    export mse
    export nmse
    export mdqrc_forecast

    export build_supervised_dataset


end