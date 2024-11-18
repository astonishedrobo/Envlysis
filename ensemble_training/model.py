import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import minimize

class LLMEnsemble:
    def __init__(self, n_models):
        self.n_models = n_models
        self.weights = None
    
    def _predictions_to_binary(self, predictions_list, unique_entities):
        """Convert list of predictions to binary vectors"""
        binary_preds = []
        for preds in predictions_list:
            binary_vec = np.zeros(len(unique_entities))
            for pred in preds:
                if pred in unique_entities:
                    binary_vec[unique_entities.index(pred)] = 1
            binary_preds.append(binary_vec)
        return np.array(binary_preds)
    
    def _objective(self, weights, X, y):
        """Objective function to minimize (1 - F1 score)"""
        weights = weights / np.sum(weights)  # Normalize weights
        weighted_pred = np.average(X, axis=0, weights=weights)
        binary_pred = (weighted_pred > 0.5).astype(int)
        return 1 - f1_score(y, binary_pred, average='macro')
    
    def fit(self, predictions_list, ground_truth):
        """
        Fit the ensemble weights
        
        Args:
            predictions_list: List of lists containing predictions from each model
            ground_truth: List containing ground truth entities
        """
        # Get unique entities across all predictions and ground truth
        unique_entities = list(set(
            [item for sublist in predictions_list + [ground_truth] 
             for item in sublist]
        ))
        
        # Convert predictions and ground truth to binary format
        X = self._predictions_to_binary(predictions_list, unique_entities)
        y = self._predictions_to_binary([ground_truth], unique_entities)[0]
        
        # Initial weights (equal weighting)
        initial_weights = np.ones(self.n_models) / self.n_models
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        bounds = [(0, 1) for _ in range(self.n_models)]
        
        # Optimize weights
        result = minimize(
            self._objective,
            initial_weights,
            args=(X, y),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        self.weights = result.x / np.sum(result.x)
        return self.weights
    
    def predict(self, predictions_list, unique_entities):
        """
        Make predictions using the ensemble
        
        Args:
            predictions_list: List of lists containing predictions from each model
            unique_entities: List of all possible entities
        """
        if self.weights is None:
            raise ValueError("Model needs to be fitted first")
            
        X = self._predictions_to_binary(predictions_list, unique_entities)
        weighted_pred = np.average(X, axis=0, weights=self.weights)
        binary_pred = (weighted_pred > 0.5).astype(int)
        
        # Convert binary predictions back to entities
        return [unique_entities[i] for i, val in enumerate(binary_pred) if val == 1]
