import re
import random
from collections import defaultdict

class TweetKMeans:
    def __init__(self, k=5, max_iterations=100, random_seed=42):
        """
        Initialize K-means clustering for tweets
       
        Args:
            k: Number of clusters
            max_iterations: Maximum iterations for convergence
            random_seed: Random seed for reproducibility
        """
        self.k = k
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.centroids = []
        self.clusters = defaultdict(list)
       
    def preprocess_tweet(self, tweet):
        """
        Preprocess a single tweet according to assignment requirements
       
        Args:
            tweet: Raw tweet string
           
        Returns:
            Set of preprocessed words
        """
        # Remove tweet id and timestamp (first two fields separated by |)
        parts = tweet.split('|')
        if len(parts) >= 3:
            tweet_text = '|'.join(parts[2:])
        else:
            tweet_text = tweet
           
        # Remove URLs
        tweet_text = re.sub(r'http\S+|www.\S+', '', tweet_text)
       
        # Remove @mentions
        tweet_text = re.sub(r'@\w+', '', tweet_text)
       
        # Remove hashtag symbols but keep the word
        tweet_text = re.sub(r'#', '', tweet_text)
       
        # Convert to lowercase
        tweet_text = tweet_text.lower()
       
        # Remove special characters and split into words
        words = re.findall(r'\b\w+\b', tweet_text)
       
        # Return as set (unordered collection of unique words)
        return set(words)
   
    def jaccard_distance(self, set_a, set_b):
        """
        Calculate Jaccard Distance between two sets
       
        Args:
            set_a: First set of words
            set_b: Second set of words
           
        Returns:
            Jaccard distance (0 = identical, 1 = completely different)
        """
        if len(set_a) == 0 and len(set_b) == 0:
            return 0.0
       
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
       
        if union == 0:
            return 1.0
           
        return 1.0 - (intersection / union)
   
    def find_centroid(self, tweet_sets):
        """
        Find the centroid of a cluster (tweet with minimum total distance to all others)
       
        Args:
            tweet_sets: List of tweet sets in the cluster
           
        Returns:
            Index of the centroid tweet
        """
        if len(tweet_sets) == 0:
            return None
        if len(tweet_sets) == 1:
            return 0
           
        min_total_distance = float('inf')
        centroid_idx = 0
       
        for i, tweet_i in enumerate(tweet_sets):
            total_distance = sum(
                self.jaccard_distance(tweet_i, tweet_j)
                for j, tweet_j in enumerate(tweet_sets) if i != j
            )
           
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                centroid_idx = i
               
        return centroid_idx
   
    def initialize_centroids(self, tweet_sets):
        """
        Initialize centroids randomly from the dataset
       
        Args:
            tweet_sets: List of all preprocessed tweets
        """
        random.seed(self.random_seed)
        self.centroids = random.sample(range(len(tweet_sets)), self.k)
   
    def assign_clusters(self, tweet_sets):
        """
        Assign each tweet to the nearest centroid
       
        Args:
            tweet_sets: List of all preprocessed tweets
           
        Returns:
            Dictionary mapping cluster_id to list of tweet indices
        """
        clusters = defaultdict(list)
       
        for idx, tweet in enumerate(tweet_sets):
            min_distance = float('inf')
            assigned_cluster = 0
           
            for cluster_id in range(self.k):
                centroid_tweet = tweet_sets[self.centroids[cluster_id]]
                distance = self.jaccard_distance(tweet, centroid_tweet)
               
                if distance < min_distance:
                    min_distance = distance
                    assigned_cluster = cluster_id
           
            clusters[assigned_cluster].append(idx)
       
        return clusters
   
    def update_centroids(self, tweet_sets, clusters):
        """
        Update centroids based on current cluster assignments
       
        Args:
            tweet_sets: List of all preprocessed tweets
            clusters: Current cluster assignments
           
        Returns:
            Boolean indicating if centroids changed
        """
        new_centroids = []
        changed = False
       
        for cluster_id in range(self.k):
            if cluster_id not in clusters or len(clusters[cluster_id]) == 0:
                # Keep old centroid if cluster is empty
                new_centroids.append(self.centroids[cluster_id])
                continue
           
            cluster_tweets = [tweet_sets[idx] for idx in clusters[cluster_id]]
            centroid_idx = self.find_centroid(cluster_tweets)
            new_centroid = clusters[cluster_id][centroid_idx]
            new_centroids.append(new_centroid)
           
            if new_centroid != self.centroids[cluster_id]:
                changed = True
       
        self.centroids = new_centroids
        return changed
   
    def calculate_sse(self, tweet_sets, clusters):
        """
        Calculate Sum of Squared Errors (SSE)
       
        Args:
            tweet_sets: List of all preprocessed tweets
            clusters: Current cluster assignments
           
        Returns:
            SSE value
        """
        sse = 0.0
       
        for cluster_id in range(self.k):
            if cluster_id not in clusters:
                continue
               
            centroid_tweet = tweet_sets[self.centroids[cluster_id]]
           
            for tweet_idx in clusters[cluster_id]:
                tweet = tweet_sets[tweet_idx]
                distance = self.jaccard_distance(tweet, centroid_tweet)
                sse += distance ** 2
       
        return sse
   
    def fit(self, tweet_sets):
        """
        Perform K-means clustering
       
        Args:
            tweet_sets: List of preprocessed tweets (sets of words)
           
        Returns:
            Tuple of (clusters, SSE, iterations)
        """
        # Initialize centroids
        self.initialize_centroids(tweet_sets)
       
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iterations):
            # Assign tweets to clusters
            clusters = self.assign_clusters(tweet_sets)
           
            # Update centroids
            changed = self.update_centroids(tweet_sets, clusters)
           
            # Check for convergence
            if not changed:
                print(f"Converged after {iteration + 1} iterations")
                break
       
        # Calculate final SSE
        sse = self.calculate_sse(tweet_sets, clusters)
       
        self.clusters = clusters
        return clusters, sse, iteration + 1


def load_and_preprocess_tweets(filename):
    """
    Load tweets from file and preprocess them
   
    Args:
        filename: Path to the tweet file
       
    Returns:
        Tuple of (raw_tweets, preprocessed_tweet_sets)
    """
    raw_tweets = []
    tweet_sets = []
   
    kmeans = TweetKMeans()  # Just for preprocessing
   
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_tweets.append(line)
                preprocessed = kmeans.preprocess_tweet(line)
                if len(preprocessed) > 0:  # Only keep non-empty tweets
                    tweet_sets.append(preprocessed)
   
    return raw_tweets, tweet_sets


def run_kmeans_experiment(tweet_sets, k_values, num_runs=5):
    """
    Run K-means clustering for multiple K values
   
    Args:
        tweet_sets: List of preprocessed tweets
        k_values: List of K values to try
        num_runs: Number of runs per K value (to handle randomness)
       
    Returns:
        Dictionary with results for each K
    """
    results = {}
   
    for k in k_values:
        print(f"\n")
        print(f"Running K-means with K={k}")
        print(f"\n")
       
        best_sse = float('inf')
        best_clusters = None
        best_run = 0
       
        # Run multiple times and keep best result
        for run in range(num_runs):
            kmeans = TweetKMeans(k=k, random_seed=42+run)
            clusters, sse, iterations = kmeans.fit(tweet_sets)
           
            print(f"Run {run+1}: SSE = {sse:.4f}, Iterations = {iterations}")
           
            if sse < best_sse:
                best_sse = sse
                best_clusters = clusters
                best_run = run + 1
       
        # Store results
        cluster_sizes = {cluster_id: len(tweets)
                        for cluster_id, tweets in best_clusters.items()}
       
        results[k] = {
            'sse': best_sse,
            'cluster_sizes': cluster_sizes,
            'best_run': best_run
        }
       
        print(f"\nBest SSE for K={k}: {best_sse:.4f} (Run {best_run})")
   
    return results


def print_results_table(results):
    """
    Prints results in formatted table for output file 
   
    Args:
        results: Dictionary with clustering results
    """
    print(f"\n")
    print("FINAL RESULTS TABLE")
    print(f"\n")
    print(f"{'Value of K':<15} {'SSE':<20} {'Size of each cluster'}")
    print("-"*70)
   
    for k in sorted(results.keys()):
        sse = results[k]['sse']
        cluster_sizes = results[k]['cluster_sizes']
       
        # Format cluster sizes
        size_str = ", ".join([f"{cid}: {size} tweets"
                             for cid, size in sorted(cluster_sizes.items())])
       
        print(f"{k:<15} {sse:<20.4f} {size_str}")
   
    


# Main execution
if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "usnewshealth.txt"  # Change this to your file
    K_VALUES = [5, 10, 15, 20, 25]   # Try at least 5 different K values
    NUM_RUNS = 5                      # Number of runs per K to handle randomness
   
    print("Loading and preprocessing tweets...")
    raw_tweets, tweet_sets = load_and_preprocess_tweets(INPUT_FILE)
   
    print(f"\nLoaded {len(tweet_sets)} valid tweets")
    print(f"Sample preprocessed tweet: {list(tweet_sets[0])}")
   
    # Run experiments
    results = run_kmeans_experiment(tweet_sets, K_VALUES, NUM_RUNS)
   
    # Print final results table
    print_results_table(results)
   
    # Optionally save results to file
    with open("results.txt", "w") as f:
        f.write("K-means Clustering Results\n")
        f.write("\n")
        f.write(f"{'Value of K':<15} {'SSE':<20} {'Size of each cluster'}\n")
        f.write("-"*70 + "\n")
       
        for k in sorted(results.keys()):
            sse = results[k]['sse']
            cluster_sizes = results[k]['cluster_sizes']
            size_str = ", ".join([f"{cid}: {size} tweets"
                                 for cid, size in sorted(cluster_sizes.items())])
            f.write(f"{k:<15} {sse:<20.4f} {size_str}\n")
   
    print("\nResults saved to 'results.txt'")