import numpy as np
from collections import defaultdict

class PersonalizedDummyClassifier:
    def __init__(self, random_state=42):
        self.class_counts = defaultdict(int)
        self.total_count = 0
        self.class_probs = {}
        self.random_state = random_state
        
        # Set the random state if provided
        self.rng = np.random.default_rng(random_state)
    
    def partial_fit(self, X, y, classes=None):
        """Update class counts based on new data."""
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            self.class_counts[cls] += count
        
        # Update total sample count
        self.total_count += len(y)
        
        # Calculate class probabilities (for stratified strategy)
        self.class_probs = {cls: count / self.total_count for cls, count in self.class_counts.items()}
    
    def predict(self, X):
        """Make predictions based on the stratified strategy."""
        n_samples = len(X)
        classes = list(self.class_probs.keys())
        probs = list(self.class_probs.values())
        
        # Predict based on the class distribution (stratified)
        return self.rng.choice(classes, size=n_samples, p=probs)

# Example usage:
