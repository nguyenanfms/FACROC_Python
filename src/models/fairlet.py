import numpy as np

class FairletClusteringScratch:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None

    def fit(self, X, P):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        idx_0 = np.where(P == 0)[0]
        idx_1 = np.where(P == 1)[0]
        
        if len(idx_0) <= len(idx_1):
            A_indices, B_indices = idx_0, idx_1
        else:
            A_indices, B_indices = idx_1, idx_0
            
        n_A = len(A_indices)
        n_B = len(B_indices)
        
        L = n_B // n_A
        R = n_B % n_A
        
        diff = X[A_indices][:, np.newaxis, :] - X[B_indices]
        dist_matrix = np.linalg.norm(diff, axis=2)
        
        b_matched = np.zeros(n_B, dtype=bool)
        fairlets = []
        
        for i in range(n_A):
            a_idx = A_indices[i]
            num_b = L + 1 if i < R else L
            
            dists = dist_matrix[i].copy()
            dists[b_matched] = np.inf
            
            closest_b_subindices = np.argsort(dists)[:num_b]
            b_matched[closest_b_subindices] = True
            
            fairlet_members = [a_idx] + list(B_indices[closest_b_subindices])
            fairlets.append(fairlet_members)
            
        fairlet_centroids = np.zeros((n_A, n_features))
        fairlet_weights = np.zeros(n_A)
        
        for j in range(n_A):
            members = fairlets[j]
            fairlet_centroids[j] = X[members].mean(axis=0)
            fairlet_weights[j] = len(members)
            
        random_indices = np.random.choice(n_A, self.n_clusters, replace=False)
        self.centroids = fairlet_centroids[random_indices].copy()
        
        fairlet_labels = np.zeros(n_A, dtype=int)
        
        for i in range(self.max_iter):
            distances = np.linalg.norm(fairlet_centroids[:, np.newaxis] - self.centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                cluster_mask = (new_labels == k)
                if np.any(cluster_mask):
                    weights = fairlet_weights[cluster_mask][:, np.newaxis]
                    points = fairlet_centroids[cluster_mask]
                    new_centroids[k] = np.sum(points * weights, axis=0) / np.sum(weights)
                else:
                    new_centroids[k] = fairlet_centroids[np.random.choice(n_A)]
                    
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                self.centroids = new_centroids
                fairlet_labels = new_labels
                break
                
            self.centroids = new_centroids
            fairlet_labels = new_labels
            
        self.labels = np.zeros(n_samples, dtype=int)
        for j in range(n_A):
            label = fairlet_labels[j]
            for member in fairlets[j]:
                self.labels[member] = label
                
        final_distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        min_distances = np.min(final_distances, axis=1)
        self.inertia = np.sum(min_distances ** 2)
        
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
