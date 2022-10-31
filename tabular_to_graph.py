"""
Created on Wed Oct 19 2022

@author: P Khn
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal


class TabGraph(object):
    def __init__(self, path) -> None:
        self.path = path
        self.path_availability = False
        self.check_path()

    # Check path availability
    def check_path(self):
        try:
            if Path(self.path).is_file():
                self.path_availability = True
        except FileNotFoundError:
            sys.exit('File does not exist!!!')

    # Get the dataset
    def get_data(self):
        if self.path_availability:
            self.data = pd.read_csv(self.path,
                                    index_col=None,
                                    dtype={'Name': str, 'Value': float})

    # Drop extra columns
    def drop_it(self):
        columns_to_drop = ['unit_number', 'time_in_cycles',
                           'op_setting_1', 'op_setting_2', 'op_setting_3']
        self.data.drop(columns_to_drop, axis=1, inplace=True)

    # Make relationship-based graph
    def make_graph(self, lag=4):
        data_adjcy_matx = np.exp(self.data.corr())
        G = nx.from_pandas_adjacency(data_adjcy_matx)
        edge_data = nx.to_pandas_edgelist(G)

        self.edge_coo = edge_data.iloc[:, 0:2].to_numpy(dtype='int64').T
        self.weights = edge_data['weight'].to_numpy()
        self.data = self.data.values
        self.features = [
            self.data[i: i + lag, :].T
            for i in range(self.data.shape[0] - lag)
        ]
        self.targets = [
            self.data[i + lag, :].T
            for i in range(self.data.shape[0] - lag)
        ]

    # Get the graph dataset
    def get_graph(self):
        dataset = StaticGraphTemporalSignal(
            self.edge_coo,
            self.weights,
            self.features,
            self.targets
        )
        return dataset




