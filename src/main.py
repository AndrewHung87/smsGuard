from utils import load_combined_data
from preprocess import preprocess_dataframe
from feature_extraction import vectorize_messages
from cluster_analysis import cluster_messages
from train_model import train_and_evaluate, save_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    # Load and preprocess the data
    df = load_combined_data('data/raw/SMSSpamCollection', 'data/raw/spam.csv')
    df = preprocess_dataframe(df)

    # Vectorize messages
    X, vectorizer = vectorize_messages(df['cleaned'])
    y = df['label']

    # Perform clustering
    kmeans = cluster_messages(X, n_clusters=2)
    cluster_labels = kmeans.labels_

    # Visualize the clustering results
    print("Visualizing Clustering Results...")
    visualize_clusters(X, cluster_labels)

    # Train and evaluate model
    model = train_and_evaluate(X, y)

    # Save the model and vectorizer
    save_model(model, vectorizer)

def visualize_clusters(X, labels):
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X.toarray())  # Convert sparse matrix to dense for PCA

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=10)
    
    # Cluster Visualization
    plt.title('Spam vs. Ham Patterns')

    # PCA Component 1
    plt.xlabel('Spam-Like Features')

    # PCA Component 2
    plt.ylabel('Message Length/Style Features')

    # Cluster Label
    plt.colorbar(label='Assigned by K-Means')
    plt.show()

if __name__ == "__main__":
    main()
