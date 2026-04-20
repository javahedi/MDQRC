"""
Core abstract types for MDQRC.

These define the main extensibility points of the framework.
New models, evolution methods, and observables should subtype these.
"""

# A quantum system (Hamiltonian + structure)
abstract type AbstractQuantumModel end

# Time evolution method (RK4, Krylov, ED, etc.)
abstract type AbstractEvolutionMethod end

# Measurement / observable definition
abstract type AbstractObservable end