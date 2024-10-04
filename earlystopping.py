import torch
import numpy as nn
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    def __call__(self, val_loss):
        if val_loss < self.best_loss:  # Check if the current loss is lower than the best loss
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if we improve
            if self.verbose:
                print(f'Validation loss improved: {val_loss:.6f}.  Saving model...')
        else:
            self.counter += 1  # Increment counter if we don't improve
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # Set the flag to stop training