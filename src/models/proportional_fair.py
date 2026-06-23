import numpy as np

class ProportionallyFairClusteringScratch:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        k = self.n_clusters
        threshold_size = int(np.ceil(n_samples / k))
        
        centers = []
        captured = np.zeros(n_samples, dtype=bool)
        
        X_norms = np.sum(X**2, axis=1)
        dist_matrix = np.sqrt(np.abs(X_norms[:, np.newaxis] + X_norms - 2 * np.dot(X, X.T)))
        
        for _ in range(k):
            if np.all(captured):
                break
                
            best_y = -1
            min_delta = np.inf
            
            active_indices = np.where(~captured)[0]
            if len(active_indices) < threshold_size:
                for y in range(n_samples):
                    if y in centers:
                        continue
                    max_d = np.max(dist_matrix[y, active_indices])
                    if max_d < min_delta:
                        min_delta = max_d
                        best_y = y
            else:
                for y in range(n_samples):
                    if y in centers:
                        continue
                    active_dists = dist_matrix[y, active_indices]
                    sorted_dists = np.partition(active_dists, threshold_size - 1)
                    delta = sorted_dists[threshold_size - 1]
                    
                    if delta < min_delta:
                        min_delta = delta
                        best_y = y
                        
            if best_y == -1:
                break
                
            centers.append(best_y)
            active_dists = dist_matrix[best_y]
            captured_mask = (~captured) & (active_dists <= min_delta + 1e-9)
            captured[captured_mask] = True
            
        while len(centers) < k:
            current_centers = np.array(centers)
            if len(centers) == 0:
                next_center = np.random.choice(n_samples)
            else:
                dists_to_centers = dist_matrix[current_centers]
                min_dists = np.min(dists_to_centers, axis=0)
                next_center = np.argmax(min_dists)
            centers.append(next_center)
            
        self.centroids = X[centers].copy()
        
        distances_to_centers = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        self.labels = np.argmin(distances_to_centers, axis=1)
        
        min_distances = np.min(distances_to_centers, axis=1)
        self.inertia = np.sum(min_distances ** 2)
        
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
