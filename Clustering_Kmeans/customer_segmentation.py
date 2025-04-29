import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import matplotlib

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

matplotlib.use('TkAgg')  # Optional for interactive windowing


def run_clustering():
    # Load dataset
    df = pd.read_csv("Mall_Customers.csv")

    # Drop ID and convert categorical to numeric
    df = df.drop("CustomerID", axis=1)
    df = pd.get_dummies(df, drop_first=True)

    # Features for clustering
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method to determine optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()

    # Final KMeans model with chosen K
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Visualization
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                    hue='Cluster', palette='viridis', s=100)
    plt.title('Customer Segments')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run_clustering()
