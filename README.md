# Customer Segmentation

The provided code performs a k-means clustering analysis on a customer dataset. Here's a breakdown of the code:

1. **Importing Libraries:**
   - `numpy` (as `np`): NumPy is used for numerical operations.
   - `pandas` (as `pd`): Pandas is used for data manipulation and analysis.
   - `matplotlib.pyplot` (as `plt`): Matplotlib is used for creating visualizations.
   - `seaborn` (as `sns`): Seaborn is a data visualization library based on Matplotlib, providing a high-level interface for drawing attractive statistical graphics.
   - `train_test_split` and `accuracy_score` from `sklearn.model_selection` and `sklearn.metrics`, respectively: These are used for splitting the data into training and testing sets and evaluating the accuracy of the clustering.

2. **Loading the Dataset:**
   - Reads a CSV file named 'customers.csv' into a Pandas DataFrame (`customer_dataset`).

3. **Exploratory Data Analysis (EDA):**
   - Displays the first few rows of the dataset using `customer_dataset.head()`.
   - Prints the shape of the dataset using `customer_dataset.shape`.
   - Checks for missing values in the dataset using `customer_dataset.isnull().sum()`.

4. **Data Preparation:**
   - Selects two columns (presumably representing features) from the dataset (`X = customer_dataset.iloc[:,[3,4]].values`).

5. **Determining the Number of Clusters (Elbow Method):**
   - Applies the k-means algorithm for different numbers of clusters (1 to 10) and calculates the within-cluster sum of squares (WCSS) for each. The results are stored in a list (`wcss`).
   - Plots the Elbow Point Graph using Seaborn and Matplotlib to help visually identify the optimal number of clusters.

6. **K-Means Clustering:**
   - Initializes a k-means model with the chosen number of clusters based on the Elbow Method (in this case, 5 clusters).
   - Fits the model to the data and assigns cluster labels to each data point (`Y = kmeans.fit_predict(X)`).

7. **Visualizing Clusters:**
   - Plots a scatter plot of the data points with different colors for each cluster.
   - Also plots the cluster centroids in black.

The code helps in understanding the structure of the data by grouping similar customer data points into clusters using the k-means algorithm. The Elbow Method is employed to find a suitable number of clusters, and the final clustering result is visualized for interpretation.
