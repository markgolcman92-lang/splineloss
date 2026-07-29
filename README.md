# Spline Loss

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A meta-learned adaptive loss function that reshapes the optimization landscape based on training dynamics.

## Core Idea

Standard loss functions (MSE, MAE) are static — they remain the same throughout training. **Spline Loss** is dynamic: a small meta-network learns to reshape the loss landscape in real-time, based on the history of prediction errors.


The meta-network outputs control points `C` of a B-spline curve, which defines how each per-sample error is mapped to a loss value.

## How It Works

1. **Main model** (DNN) predicts outputs
2. **Meta-network** observes the history of training losses
3. Meta-network outputs control points `C` of a B-spline
4. B-spline maps per-sample MSE to a shaped loss value
5. Gradient flows through the shaped loss to update the main model
6. Meta-network is updated via a meta-objective on validation data

## Why It Works

The meta-network learns to:
- **Amplify gradients** when the model is stuck (error stagnates)
- **Smooth gradients** when the model converges
- **Build robustness** to outliers via the spline's flexible shape

## Key Properties

| Property | Description |
|----------|-------------|
| **Adaptive** | Loss shape changes during training |
| **Interpretable** | Control points show the learned loss shape |
| **Compatible** | Works with any optimizer (SGD, Adam, AdamW) |
| **Robust** | B-spline is flexible and stable |

## Example: Learned Loss Shape

The meta-network learns different loss shapes for different training phases:

| Phase | Loss Shape | Purpose |
|-------|------------|---------|
| **Early** | Gentle slope | Stabilize training |
| **Mid** | Steep in high-error region | Push hard on mistakes |
| **Late** | Flattened | Fine-tune without overreacting |


