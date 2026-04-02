# Non-Invertibility and Emptiness

This repository contains the code and experiments accompanying the paper:

**"Non-Invertibility and Emptiness: A Formal Reconstruction of the Madhyamaka Analysis of Identity"**

---

## Overview

This project explores the relationship between:

- Non-invertibility in generative models
- Structural non-identifiability
- The absence of inherent identity (Madhyamaka philosophy)

We show that:

- Multiple latent configurations can generate the same observable structure
- No uniquely identifiable underlying state exists
- The space of compatible solutions is highly constrained and low-dimensional

---

## Key Result

The solution space is not arbitrary. It is:

> **effectively low-dimensional, dominated by a single principal mode**

This provides a formal analogue of the Madhyamaka concept of emptiness:

- Not absence of structure  
- But absence of inherent, independent identity  

---

## Repository Structure

- `experiments/` — numerical experiments (Exp 1–9)
- `figures/` — figures used in the paper
- `paper/` — PDF of the paper

---
## Code Structure

The codebase is intentionally kept simple and experiment-driven.

Each script is self-contained and corresponds directly to a specific experiment in the paper. This avoids unnecessary abstraction and preserves transparency.

---
Gonzalez-Granda Fernandez, E. (2026).
Non-Invertibility and Emptiness: A Formal Reconstruction of the Madhyamaka Analysis of Identity.
Zenodo. DOI:10.5281/zenodo.19376217
## How to Run

Example:

```bash
python experiments/exp9_intrinsic_dimension.py
