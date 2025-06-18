import numpy as np

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def cluster_points(points, eps, min_samples):    
    # points - Массив точек
    # eps - Радиус окрестности для определения близости
    # min_samples - Минимальное количество точек для формирования кластера
    clusters = []
    visited = set()
    for i, point in enumerate(points):
        if i in visited:
            continue
        cluster = []
        queue = [i]
        while queue:
            idx = queue.pop(0)
            if idx in visited:
                continue
            visited.add(idx)
            cluster.append(idx)
            for j, other_point in enumerate(points):
                if j not in visited and distance(points[idx], other_point) < eps:
                    queue.append(j)
        if len(cluster) >= min_samples:
            clusters.append(cluster)
    return clusters

def compute_centroids(clusters, points):
    centroids = []
    for cluster in clusters:
        cluster_points = np.array([points[i] for i in cluster])
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return centroids