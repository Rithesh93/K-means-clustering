import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1,x2):
   return np.sqrt(np.sum((x1-x2)**2))

class KMC():
    
    def __init__(self, K=5, n_iters=100, plot_steps=False):
        self.K = K
        self.n_iters = n_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]           #creating an empty cluster
        # the centers (mean feature vector) for each cluster
        self.centroids = []     #creating an empty list of centroids

        
    def predict(self,X):
        self.X=X
        self.n_samples,self.n_features=X.shape
        #initialize centroids
        centroids_idxs=np.random.choice(self.n_samples,self.K,replace=False)    #randomly initializing the centroids
        self.centroids=[self.X[i] for i in centroids_idxs]
        #create clusters
        for _ in range(self.n_iters):
            self.clusters=self._create_clusters(self.centroids)        #calling the function of creating clusters usinmg centroids
            if self.plot_steps:
                self.plot()
            #calculate new centroids from the clusters
            old_centroids=self.centroids
            self.centroids=self.get_centroids(self.clusters)    #getting updated centroids by averaging
            if self._is_converged(old_centroids,self.centroids):   #checking to see if the sum of the distances between new and old centroids have converged to zero
                break
            if self.plot_steps:
                self.plot()
        return self.get_labels(self.clusters)
    
    def get_labels(self,clusters):
        labels=np.empty(self.n_samples)
        for cluster_idx,cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx]=cluster_idx 
            
    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index
    
    def get_centroids(self,clusters):
        centroids=np.zeros((self.K,self.n_features))
        for idx, cluster in enumerate(clusters):
            cluster_mean=np.mean(self.X[cluster],axis=0)
            centroids[idx]=cluster_mean
        return centroids
    
    def _is_converged(self,old_centroids,new_centroids):
        convergence=[euclidean_distance(old_centroids[i],new_centroids[i]) for i in range(self.K)]
        return sum(convergence)==0
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)
        plt.show()
