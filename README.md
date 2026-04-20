# MDQRC.jl

[![Build Status](https://github.com/javahedi/MDQRC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/javahedi/MDQRC.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Measurement-Driven Quantum Reservoir Computing (MD-QRC)** in Julia.

---

## 🚀 Overview

**MD-QRC** is a framework for time-series processing using quantum many-body systems in the **NISQ era**, designed to be:

- **hardware-realistic**
- **measurement-driven**
- **feedback-free**
- **memory-efficient**

Unlike standard Quantum Reservoir Computing (QRC), MD-QRC:

> **decouples nonlinear quantum dynamics from memory storage**

- Quantum system → nonlinear feature generator  
- Classical buffer → temporal memory  

---

## 🧠 Core Idea

At each timestep:

1. A quantum system evolves under a driven Hamiltonian
2. Local observables are measured (e.g. ⟨Zᵢ(t)⟩)
3. Measurements are stored in a classical delay buffer
4. A linear readout is trained on this embedded history

\[
\mathbf{x}(t) = [z(t), z(t-1), ..., z(t-\tau)]
\]

\[
\hat{y}(t) = W \cdot \mathbf{x}(t)
\]

---

## 🔁 Comparison with Existing QRC

| Feature | Standard QRC | PRX Feedback QRC | **MD-QRC (this work)** |
|--------|-------------|-----------------|------------------------|
| Memory | Quantum state | Feedback loop | **Classical delay embedding** |
| Measurement | Final | Every step | **Every step (passive)** |
| Feedback | ❌ | ✅ | ❌ |
| Hardware demand | High | Very high | **Low** |
| Data reuse | ❌ | ❌ | **✅ (offline training)** |

---

## ⚙️ Features

- Matrix-free quantum evolution (no dense Hamiltonians)
- Bitstring-based state representation
- Multiple evolution methods:
  - RK4
  - Krylov subspace exponential
- Observable streaming (local measurements)
- Delay embedding (Takens-style)
- Ridge regression readout
- Batch experiment support
- Parallel execution (via `pmap`)

---

## 🧩 Project Structure

```

src/
├── types.jl
├── basis.jl
├── models/
│   └── tfim.jl
├── evolution/
│   ├── rk4.jl
│   └── krylov.jl
├── observables/
│   └── localz.jl
├── reservoir/
│   └── embedding.jl
├── readout/
│   └── ridge.jl

````

---

## 📦 Installation

```julia
using Pkg
Pkg.add(url="https://github.com/javahedi/MDQRC.jl")
````

---

## 🧪 Quick Example

```julia
using MDQRC

# System
N = 10
model = TFIMChain(N, 1.0, 1.05, -0.5, [1,2])
ψ0 = product_state(N)

# Input signal
function hz(t)
    return 0.01 * sin(0.1 * t)
end

# Observables
obs = LocalZ([1,2,3,4])

# Evolution
stream = evolve_observables(
    model,
    ψ0,
    obs;
    dt=0.05,
    steps=1000,
    hz=hz,
    method=KrylovExp(m=20),
)

data = stream.data
```

---

## 🔬 Supported Experiments

### 1. Memory Capacity

Evaluate fading memory without feedback.

### 2. Time-Series Forecasting

Predict future observables:
[
Z_i(t+\tau)
]

### 3. Cross-Prediction

Predict distant spins from local measurements:
[
[Z_1, Z_2, Z_3, Z_4] \rightarrow Z_{10}
]

### 4. Regime Analysis

* Integrable vs chaotic dynamics
* Effect of disorder
* Information spreading

---

## 📊 Key Insight

> **Prediction performance is governed by dynamical correlations, not geometric distance.**

MD-QRC reveals how local measurements encode global quantum information via operator spreading.

---

## ⚡ Design Principles

* **Matrix-free physics** → scalable Hilbert space
* **Separation of concerns**

  * dynamics (quantum)
  * memory (classical)
* **Minimal allocations**
* **Composable components**
* **Research-first architecture**

---

## 🧭 Roadmap

* [ ] Disorder and MBL studies
* [ ] Entanglement estimation
* [ ] Experimental hardware integration
* [ ] GPU acceleration
* [ ] Multi-observable kernels

---

## 📄 Citation

If you use this package, please cite:

```
Measurement-Driven Quantum Reservoir Computing (MD-QRC)
J. Vahedi et al. (in preparation)
```

---

## 🤝 Contributing

Contributions are welcome!

* open issues
* propose new models
* improve performance
* add experiments

---

## 📜 License

MIT License

```

---

# 🔥 Why this README is strong

It does 3 critical things:

### 1. Clearly positions your idea
Not just "a package" — a **new QRC paradigm**

### 2. Speaks to both audiences
- physicists → Hamiltonian, observables
- ML people → embedding, regression

### 3. Matches your actual code
No fake abstraction — everything is grounded in your implementation

---

# 🚀 Next (optional but powerful)

Later you can add:

- diagrams (very impactful)
- example notebooks
- benchmark plots


