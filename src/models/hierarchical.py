import numpy as np

class HierarchicalScratch:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels = None
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        active = np.ones(n_samples, dtype=bool)
        sizes = np.ones(n_samples, dtype=float)
        
        X_norms = np.sum(X**2, axis=1)
        dist_matrix = 0.5 * (X_norms[:, np.newaxis] + X_norms - 2 * np.dot(X, X.T))
        dist_matrix[dist_matrix < 0] = 0
        np.fill_diagonal(dist_matrix, np.inf)
        
        members = [[i] for i in range(n_samples)]
        
        for step in range(n_samples - self.n_clusters):
            u, v = divmod(np.argmin(dist_matrix), n_samples)
            
            sz_u = sizes[u]
            sz_v = sizes[v]
            
            d_u = dist_matrix[:, u].copy()
            d_v = dist_matrix[:, v].copy()
            d_uv = dist_matrix[u, v]
            
            members[u].extend(members[v])
            members[v] = []
            
            active[v] = False
            dist_matrix[v, :] = np.inf
            dist_matrix[:, v] = np.inf
            
            sizes[u] = sz_u + sz_v
            sizes[v] = 0
            
            k_mask = active.copy()
            k_mask[u] = False
            
            if np.any(k_mask):
                sz_k = sizes[k_mask]
                new_d = ((sz_u + sz_k) * d_u[k_mask] + (sz_v + sz_k) * d_v[k_mask] - sz_k * d_uv) / (sz_u + sz_v + sz_k)
                dist_matrix[k_mask, u] = new_d
                dist_matrix[u, k_mask] = new_d
                
            dist_matrix[u, u] = np.inf
            
        self.labels = np.zeros(n_samples, dtype=int)
        label_idx = 0
        for i in range(n_samples):
            if active[i]:
                for idx in members[i]:
                    self.labels[idx] = label_idx
                label_idx += 1
                
        self.centroids = np.zeros((self.n_clusters, n_features))
        for k in range(self.n_clusters):
            self.centroids[k] = X[self.labels == k].mean(axis=0)
            
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
