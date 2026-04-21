# Tabular-to-Graph

A small utility for converting tabular time-series data into a temporal graph dataset.

## Overview

`Tabular-to-Graph` transforms tabular sensor data into a graph-based representation for graph machine learning workflows. It builds a relationship graph from feature correlations, creates lag-based temporal features, and returns a dataset compatible with `torch_geometric_temporal`.

This project is based on the NASA turbofan engine degradation dataset and is intended for experiments in predictive modeling, prognostics, and graph-based time-series learning.

## Features

- Load tabular sensor data from CSV
- Drop non-sensor operating columns
- Build a correlation-based graph with weighted edges
- Generate lag-based temporal graph features
- Export data in `StaticGraphTemporalSignal` format

## Tech Stack

- Python
- Pandas
- NumPy
- NetworkX
- PyTorch Geometric Temporal

## Project Structure

```text
README.md
tabular_to_graph.py
train_FD004.csv
```

## Dataset

The repository uses the NASA turbofan engine degradation simulation dataset:

- A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Prognostics Data Repository

## How It Works

The pipeline:
1. loads the tabular dataset
2. removes selected non-sensor columns
3. computes feature correlations
4. converts the correlation matrix into a graph
5. creates lag-based temporal features and targets
6. returns a temporal graph dataset for downstream modeling


## License

MIT
