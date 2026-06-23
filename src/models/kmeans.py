import numpy as np

class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='kmeans++', n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None

    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape
        
        # 1. Khởi tạo ngẫu nhiên đơn giản
        if self.init == 'random':
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[random_indices].copy()
            
        # 2. Khởi tạo bằng thuật toán K-Means++ (Khuyên dùng)
        elif self.init == 'kmeans++':
            # Bước a: Chọn tâm cụm đầu tiên ngẫu nhiên từ tập dữ liệu
            centroids = [X[np.random.choice(n_samples)]]
            
            # Bước b: Chọn K-1 tâm cụm còn lại dựa trên xác suất bình phương khoảng cách
            for _ in range(1, self.n_clusters):
                current_centroids = np.array(centroids)
                # Tính khoảng cách từ mọi điểm đến các tâm cụm đã chọn (Vectorized)
                distances = np.linalg.norm(X[:, np.newaxis] - current_centroids, axis=2) # Shape: (N, num_chosen)
                # Tìm khoảng cách ngắn nhất từ mỗi điểm đến tâm cụm gần nhất
                min_sq_distances = np.min(distances, axis=1) ** 2
                
                # Tính xác suất chọn (điểm nào ở xa các tâm cụm cũ thì xác suất được chọn làm tâm cụm mới càng cao)
                sum_sq_dist = np.sum(min_sq_distances)
                # Đề phòng trường hợp tổng khoảng cách bằng 0 (dữ liệu trùng lặp hoàn toàn)
                probs = min_sq_distances / sum_sq_dist if sum_sq_dist > 0 else np.ones(n_samples) / n_samples
                
                # Chọn tâm cụm tiếp theo
                next_centroid_idx = np.random.choice(n_samples, p=probs)
                centroids.append(X[next_centroid_idx])
                
            return np.array(centroids)
        else:
            raise ValueError("init method must be either 'random' or 'kmeans++'")

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        
        # Tạo seed cơ sở để mỗi lần lặp chạy độc lập nhưng có tính lặp lại được
        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 100000)
        
        for run in range(self.n_init):
            np.random.seed(base_seed + run)
            
            centroids = self._initialize_centroids(X)
            labels = None
            
            for i in range(self.max_iter):
                # A. Bước gán (Assignment Step): 
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2) # Shape: (N, K)
                new_labels = np.argmin(distances, axis=1)
                
                # B. Bước cập nhật (Update Step):
                new_centroids = np.zeros_like(centroids)
                for k in range(self.n_clusters):
                    cluster_points = X[new_labels == k]
                    if len(cluster_points) > 0:
                        new_centroids[k] = cluster_points.mean(axis=0)
                    else:
                        new_centroids[k] = X[np.random.choice(n_samples)]
                
                # C. Kiểm tra hội tụ (Convergence Check):
                if np.all(np.abs(new_centroids - centroids) < self.tol):
                    centroids = new_centroids
                    labels = new_labels
                    break
                    
                centroids = new_centroids
                labels = new_labels
                
            # D. Tính toán Inertia cho lượt chạy này
            final_distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            min_distances = np.min(final_distances, axis=1)
            inertia = np.sum(min_distances ** 2)
            
            # Giữ lại kết quả tốt nhất (Inertia nhỏ nhất)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                
        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = best_inertia
        
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)