# import libraries for text processing and random number generation 
import re
import random

# import to create dictionaries with default values 
from collections import defaultdict

# used to initialize a K-means clustering 
class TweetKMeans:
    def __init__(self, k=5, max_iterations=100, random_seed=42):
        # number of clusters
        self.k = k

        # iterations
        self.max_iterations = max_iterations
        self.random_seed = random_seed

        # list to store center tweets index values
        self.centroids = []

        # dictionary to store which cluster the tweet is assigned to 
        self.clusters = defaultdict(list)
       
    # used to clean up and standardize each tweet 
    def preprocess_tweet(self, tweet):

        # remove unecessary fields (tweet id and timestamp)
        parts = tweet.split('|')

        # check if tweet has at least 3 parts, parse, and remove first two parts 
        if len(parts) >= 3:
            tweet_text = '|'.join(parts[2:])
        else:
            # let tweet be as it is 
            tweet_text = tweet
           
        # remove any URLs within the tweet 
        tweet_text = re.sub(r'http\S+|www.\S+', '', tweet_text)
       
        # remove @mentions
        tweet_text = re.sub(r'@\w+', '', tweet_text)
       
        # remove # 
        tweet_text = re.sub(r'#', '', tweet_text)
       
        # convert tweet messages to lowercase
        tweet_text = tweet_text.lower()
       
        # remove miscallaneous characters 
        words = re.findall(r'\b\w+\b', tweet_text)
       
        # return set 
        return set(words)
   
   # calculate the distance between two sets (tweets)
    def jaccard_distance(self, set_a, set_b):
        
        # both sets are empty and thus identical. return zero distance
        if len(set_a) == 0 and len(set_b) == 0:
            return 0.0
       
        # calculate number of common elements between the two sets 
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
       
       # if union is empty, max distance is 1
        if union == 0:
            return 1.0
        
        # calculate Jaccard distance and return using formula
        return 1.0 - (intersection / union)
   
   # find the center tweet of a cluster of tweets that is the closest overall to other tweets 
    def find_centroid(self, tweet_sets):

        # check if tweets list is empty 
        if len(tweet_sets) == 0:
            return None
        
        # check if there is only one tweet in the cluster 
        if len(tweet_sets) == 1:
            return 0
        
        min_total_distance = float('inf')

        # initialize center distance to 0 at first 
        centroid_idx = 0
       
       # go through all the tweets in the cluster using its index 
        for i, tweet_i in enumerate(tweet_sets):

            # find the total sum of distances from a particular tweet to all the other tweets 
            # within the cluster 
            total_distance = sum(
                # Jaccard distance calculation
                self.jaccard_distance(tweet_i, tweet_j)

                # go through all the other tweets in the cluster 
                for j, tweet_j in enumerate(tweet_sets) if i != j
            )
           
           # check if the total distance of the current tweet is less than the global minmum distance
            if total_distance < min_total_distance:

                # if it is less than global minimum, update minimum 
                min_total_distance = total_distance
                # centroid found, update its index to point to current tweet 
                centroid_idx = i
               
        return centroid_idx
   
   # randomly initialize centroids from dataset
    def initialize_centroids(self, tweet_sets):
        # set a random seed for standardization and recall through various iterations
        random.seed(self.random_seed)
        # sample k indices and store as centroids
        self.centroids = random.sample(range(len(tweet_sets)), self.k)
   
   # assign each tweet to the nearest cluster + centroid tweet 
    def assign_clusters(self, tweet_sets):
        # store cluster assignments 
        clusters = defaultdict(list)
       
       # loop through each tweet in the dataset 
        for idx, tweet in enumerate(tweet_sets):
            min_distance = float('inf')
            assigned_cluster = 0
           
           # loop through each of the clusters 
            for cluster_id in range(self.k):
                # get the centroid tweet within the cluster 
                centroid_tweet = tweet_sets[self.centroids[cluster_id]]
                
                # calculate the Jaccard distance between tweet and the centroid tweet 
                distance = self.jaccard_distance(tweet, centroid_tweet)
               
               # verify if the distance to the centroid is less than the global minimum 
                if distance < min_distance:
                    # update minimum distance and assigned cluster
                    min_distance = distance
                    assigned_cluster = cluster_id
           
           # add the index value of the tweet list of tweets in its assigned cluster
           # to keep track of which tweets belong to each cluster
            clusters[assigned_cluster].append(idx)
       
       # return mapping between tweet index values and assigned cluster 
        return clusters
   
   # update centroid assignment based on current cluster assignment of all the tweets 
    def update_centroids(self, tweet_sets, clusters):

        # intialize list to easily access centroid index values 
        new_centroids = []

        # track if any centroid has been updated or changed
        changed = False
       
       # go through each cluster 
        for cluster_id in range(self.k):
            if cluster_id not in clusters or len(clusters[cluster_id]) == 0:
                # if cluster is empty, retain old centroid as the updated centroid 
                new_centroids.append(self.centroids[cluster_id])
                continue
           
           # get all the tweets in the current cluster 
            cluster_tweets = [tweet_sets[idx] for idx in clusters[cluster_id]]

            # find index of centroid that has the least overall distance to other tweets 
            centroid_idx = self.find_centroid(cluster_tweets)

            # get tweet index of the centroid 
            new_centroid = clusters[cluster_id][centroid_idx]

            # add new centroid to centroid list 
            new_centroids.append(new_centroid)
           
           # verify if new centroid is = to old centroid 
            if new_centroid != self.centroids[cluster_id]:
                # flag is now true to indicate a centroid has been changed
                changed = True
       
       # update metdata with new centroid indices 
        self.centroids = new_centroids

        # return if any centroids have been changed 
        return changed
   
   # calculate SSE (error metric)
    def calculate_sse(self, tweet_sets, clusters):
        sse = 0.0
       
       # loop through clusters 
        for cluster_id in range(self.k):
            # check if a cluster is not in the dictionary 
            if cluster_id not in clusters:
                continue
               
            # get the tweet htat is the center of the current cluster 
            centroid_tweet = tweet_sets[self.centroids[cluster_id]]
           
           # loop through all the tweets within the cluster 
            for tweet_idx in clusters[cluster_id]:
                # get tweet set
                tweet = tweet_sets[tweet_idx]

                # calculate Jacard distance between tweet and centroid
                distance = self.jaccard_distance(tweet, centroid_tweet)
                
                # calulate SSE
                sse += distance ** 2
        # return SSE
        return sse
   
   # complete K-means clustering mechanism
    def fit(self, tweet_sets):
        # initialize centroids by randomly selecting tweets within the dataset 
        self.initialize_centroids(tweet_sets)
       
        # iterate from 0 to max iterations 
        for iteration in range(self.max_iterations):
            # assign each tweet to nearest cluster + centroid 
            clusters = self.assign_clusters(tweet_sets)
           
            # update centroids based on current cluster 
            changed = self.update_centroids(tweet_sets, clusters)
           
            # check if no centroids have been changed
            if not changed:
                print(f"Converged after {iteration + 1} iterations")
                break
       
        # calculate final SSE metric
        sse = self.calculate_sse(tweet_sets, clusters)
       
       # store cluster assignments 
        self.clusters = clusters

        # return metadata about the process of K-means
        return clusters, sse, iteration + 1


# take tweets from the file and preprocess them in prep for k-means clustering
def load_and_preprocess_tweets(filename):
    # initialize list to store tweet contents 
    raw_tweets = []

    # initialize list to store cleaned up tweets 
    tweet_sets = []
   
   # access preprocessing method 
    kmeans = TweetKMeans()  
   
   # open file to read tweets 
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        # loop through each lines 
        for line in f:
            # remove whitespace for standardization from lines
            line = line.strip()

            # check if line contains content 
            if line:
                # add tweet body to raw tweets list 
                raw_tweets.append(line)

                # preprocess the tweet to perform K-means
                preprocessed = kmeans.preprocess_tweet(line)

                # check if preprocessed tweet still contains content
                if len(preprocessed) > 0:  
                    tweet_sets.append(preprocessed)
   # return both sets of tweets (processed and not preprocessed)
    return raw_tweets, tweet_sets

# perform k-means clustering for various values of k clusters 
def run_kmeans_experiment(tweet_sets, k_values, num_runs=5):
    # empty dictionary to store results 
    results = {}
   
   # loop through to show progress of each k value in list 
    for k in k_values:
        print(f"\n")
        print(f"Running K-means with K={k}")
        print(f"\n")
       
       # initialize variables to check what the best value of k is for the dataset
        best_sse = float('inf')
        best_clusters = None
        best_run = 0
       
        # complete various trials and  store the best outcome
        for run in range(num_runs):
            # create new instance with current value of k
            kmeans = TweetKMeans(k=k, random_seed=42+run)

            # run k-means clustering 
            clusters, sse, iterations = kmeans.fit(tweet_sets)
           
           # print results for the current iteration
            print(f"Run {run+1}: SSE = {sse:.4f}, Iterations = {iterations}")
           
           # check for the best SSE within the iterations 
            if sse < best_sse:
                best_sse = sse
                best_clusters = clusters
                best_run = run + 1
       
        # store results in the dictionary
        cluster_sizes = {cluster_id: len(tweets)
                        for cluster_id, tweets in best_clusters.items()}
       
        results[k] = {
            'sse': best_sse,
            'cluster_sizes': cluster_sizes,
            'best_run': best_run
        }
       
       # print summary statement displaying best SSE for the current value of k
        print(f"\nBest SSE for K={k}: {best_sse:.4f} (Run {best_run})")
   
   # return dictionary containing results for all k values
    return results

# print results in a table within an output file 
def print_results_table(results):
    # print with proper foramtting
    print(f"\n")
    print("FINAL RESULTS TABLE")
    print(f"\n")
    print(f"{'Value of K':<15} {'SSE':<20} {'Size of each cluster'}")
    print("-"*70)
   
   # go through all the k values 
    for k in sorted(results.keys()):
        # get SSE 
        sse = results[k]['sse']
        cluster_sizes = results[k]['cluster_sizes']
       
        # format the size of clusters
        size_str = ", ".join([f"{cid}: {size} tweets"
                             for cid, size in sorted(cluster_sizes.items())])
       
        print(f"{k:<15} {sse:<20.4f} {size_str}")
   
    


# main driver of the progra
if __name__ == "__main__":
    INPUT_FILE = "usnewshealth.txt"  
    K_VALUES = [5, 10, 15, 20, 25]   # try various values of k
    NUM_RUNS = 5                      # number of runs per k
   
   # progress message to show tweets are being cleaned up 
    print("Loading and preprocessing tweets...")
    raw_tweets, tweet_sets = load_and_preprocess_tweets(INPUT_FILE)
   
   # print number of valid tweets within the dataset 
    print(f"\nLoaded {len(tweet_sets)} valid tweets")
    print(f"Sample preprocessed tweet: {list(tweet_sets[0])}")
   
    # run trials of k-means
    results = run_kmeans_experiment(tweet_sets, K_VALUES, NUM_RUNS)
   
    # print results to a table for easy viewing
    print_results_table(results)
   
    # save results to an external file in proper formatting 
    with open("results.txt", "w") as f:
        f.write("K-means Clustering Results\n")
        f.write("\n")
        f.write(f"{'Value of K':<15} {'SSE':<20} {'Size of each cluster'}\n")
        f.write("-"*70 + "\n")
       
       # go through k values in order
        for k in sorted(results.keys()):

            # obtain SSE value of current k assignment from dictionary 
            sse = results[k]['sse']
            
            # get cluster sizes from current k value 
            cluster_sizes = results[k]['cluster_sizes']

            # format sizes properly
            size_str = ", ".join([f"{cid}: {size} tweets"
                                 for cid, size in sorted(cluster_sizes.items())])
            f.write(f"{k:<15} {sse:<20.4f} {size_str}\n")
   
   # print success message for results being saved
    print("\nResults saved to 'results.txt'")