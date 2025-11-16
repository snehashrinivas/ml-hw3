# Intro to ML Homework 2 - Code Portion 
## Written by Sneha Shrinivas and Khushi Dubey 

In this project we created a program that implements a tweet clustering function using the Jaccard Distance metric and a K-means clustering algorithm. 

---

### To Run: 

- Make sure that you have Python 3.6 or higher.
- From the UCI Dataset at this [link](https://archive.ics.uci.edu/dataset/438/health+news+in+twitter) download a tweet file in the form of a .txt (for example, the usnewshealth.txt)
- Place the txt file in the same repository as the program
- Open `kmeans_clustering.py` and ensure the configuration section at the bottom is as follows:

```python
# Configuration
INPUT_FILE = "usnewshealth.txt"  # Change to your tweet file
K_VALUES = [5, 10, 15, 20, 25]   # K values to try
NUM_RUNS = 5                      # Runs per K (best result selected)

```

- Run `python kmeans_clustering.py` or `python3 kmeans_clustering.py` depending on how your environment is set up
- The results will save to a 'results.txt' file in the directory.
- For best viewing results, open the results.txt file in the IDE itself to see the table properly formatted with no wrap. 

We used the `re` library for preprocessing, the `random` library for random centroid initialization, and `collections.defaultdict` for cluster management. 
