from setup_kaggle_api import setup_kaggle_api
from Clustering_Kmeans.download_dataset import download_dataset
from Clustering_Kmeans.customer_segmentation import run_clustering

# Change this to your actual path
kaggle_json_path = "C:/Users/YourName/Downloads/kaggle.json"

setup_kaggle_api(kaggle_json_path)
download_dataset()
run_clustering()
