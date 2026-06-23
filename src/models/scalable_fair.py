import numpy as np

class ScalableFairClusteringScratch:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None

    def _kd_tree_decomposition(self, X, P, indices, depth=0):
        if len(indices) == 0:
            return [], [], []
            
        n_samples = len(indices)
        idx_0 = indices[P[indices] == 0]
        idx_1 = indices[P[indices] == 1]
        
        if n_samples <= 10 or depth >= 15:
            min_len = min(len(idx_0), len(idx_1))
            fairlets = []
            for i in range(min_len):
                fairlets.append([idx_0[i], idx_1[i]])
            left_0 = list(idx_0[min_len:])
            left_1 = list(idx_1[min_len:])
            return left_0, left_1, fairlets
            
        X_subset = X[indices]
        variances = np.var(X_subset, axis=0)
        split_dim = np.argmax(variances)
        
        median_val = np.median(X_subset[:, split_dim])
        left_mask = X[indices, split_dim] <= median_val
        left_child_idx = indices[left_mask]
        right_child_idx = indices[~left_mask]
        
        if len(left_child_idx) == 0 or len(right_child_idx) == 0:
            min_len = min(len(idx_0), len(idx_1))
            fairlets = []
            for i in range(min_len):
                fairlets.append([idx_0[i], idx_1[i]])
            left_0 = list(idx_0[min_len:])
            left_1 = list(idx_1[min_len:])
            return left_0, left_1, fairlets
            
        left_0, left_1, left_fairlets = self._kd_tree_decomposition(X, P, left_child_idx, depth + 1)
        right_0, right_1, right_fairlets = self._kd_tree_decomposition(X, P, right_child_idx, depth + 1)
        
        pool_0 = left_0 + right_0
        pool_1 = left_1 + right_1
        
        min_len = min(len(pool_0), len(pool_1))
        node_fairlets = left_fairlets + right_fairlets
        
        for i in range(min_len):
            node_fairlets.append([pool_0[i], pool_1[i]])
            
        leftovers_0 = pool_0[min_len:]
        leftovers_1 = pool_1[min_len:]
        
        return leftovers_0, leftovers_1, node_fairlets

    def fit(self, X, P):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        indices = np.arange(n_samples)
        left_0, left_1, fairlets = self._kd_tree_decomposition(X, P, indices)
        
        if len(left_0) > 0:
            leftovers = left_0
        else:
            leftovers = left_1
            
        if len(leftovers) > 0 and len(fairlets) > 0:
            fairlet_centroids = np.array([X[f].mean(axis=0) for f in fairlets])
            leftover_points = X[leftovers]
            closest_fairlet_indices = []
            for lp in leftover_points:
                d = np.sum((fairlet_centroids - lp) ** 2, axis=1)
                closest_fairlet_indices.append(np.argmin(d))
            closest_fairlet_indices = np.array(closest_fairlet_indices)
            
            for i, idx in enumerate(leftovers):
                closest_f_idx = closest_fairlet_indices[i]
                fairlets[closest_f_idx].append(idx)
        elif len(fairlets) == 0:
            fairlets = [list(indices)]
            
        n_fairlets = len(fairlets)
        fairlet_centroids = np.zeros((n_fairlets, n_features))
        fairlet_weights = np.zeros(n_fairlets)
        
        for j in range(n_fairlets):
            members = fairlets[j]
            fairlet_centroids[j] = X[members].mean(axis=0)
            fairlet_weights[j] = len(members)
            
        random_indices = np.random.choice(n_fairlets, min(self.n_clusters, n_fairlets), replace=False)
        self.centroids = fairlet_centroids[random_indices].copy()
        
        fairlet_labels = np.zeros(n_fairlets, dtype=int)
        
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
                    new_centroids[k] = fairlet_centroids[np.random.choice(n_fairlets)]
                    
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                self.centroids = new_centroids
                fairlet_labels = new_labels
                break
                
            self.centroids = new_centroids
            fairlet_labels = new_labels
            
        self.labels = np.zeros(n_samples, dtype=int)
        for j in range(n_fairlets):
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
