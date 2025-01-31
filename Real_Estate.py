import pandas as pd  # Import pandas for data manipulation and analysis

# Load the real estate dataset
real_estate_data = pd.read_csv("Real_Estate.csv")

# Display the first few rows of the dataset and basic dataset information
real_estate_data_head = real_estate_data.head()  # Get the first 5 rows of the dataset
data_info = real_estate_data.info()  # Display dataset structure, column types, and missing values

print(real_estate_data_head)  # Print the first 5 rows
print(data_info)  # Print dataset information

# Check for missing values in each column
print(real_estate_data.isnull().sum())

# Generate descriptive statistics of the dataset (mean, median, standard deviation, etc.)
descriptive_stats = real_estate_data.describe()

print(descriptive_stats)  # Print descriptive statistics

import matplotlib.pyplot as plt  # Import Matplotlib for data visualization
import seaborn as sns  # Import Seaborn for enhanced visualization

# Set the aesthetic style of the plots to "whitegrid"
sns.set_style("whitegrid")

# Create histograms for numerical columns to visualize distributions
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))  # Create a 3x2 grid of subplots
fig.suptitle('Histograms of Real Estate Data', fontsize=16)  # Set the title for the figure

# List of numerical columns to visualize
cols = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores',
        'Latitude', 'Longitude', 'House price of unit area']

# Plot a histogram for each column
for i, col in enumerate(cols):
    sns.histplot(real_estate_data[col], kde=True, ax=axes[i//2, i%2])  # Plot histogram with density curve (kde=True)
    axes[i//2, i%2].set_title(col)  # Set subplot title
    axes[i//2, i%2].set_xlabel('')  # Remove X-axis label
    axes[i//2, i%2].set_ylabel('')  # Remove Y-axis label

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot layout to prevent overlapping
plt.show()  # Display the histograms

# Create scatter plots to observe relationships with house price
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))  # Create a 2x2 grid of subplots
fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16)  # Set the figure title

# Plot scatter plots to examine relationships between features and house price
sns.scatterplot(data=real_estate_data, x='House age', y='House price of unit area', ax=axes[0, 0])
sns.scatterplot(data=real_estate_data, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0, 1])
sns.scatterplot(data=real_estate_data, x='Number of convenience stores', y='House price of unit area', ax=axes[1, 0])
sns.scatterplot(data=real_estate_data, x='Latitude', y='House price of unit area', ax=axes[1, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot layout
plt.show()  # Display scatter plots

# Check data types to identify non-numeric columns
print(real_estate_data.dtypes)

# Select only numerical columns for correlation analysis
numeric_data = real_estate_data.select_dtypes(include=['number'])

# Compute the correlation matrix only for numeric columns
correlation_matrix = numeric_data.corr()

# Plot the correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

print(correlation_matrix)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Selecting features and target variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

X = real_estate_data[features]
y = real_estate_data[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions using the linear regression model
y_pred_lr = model.predict(X_test)

# Visualization: Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted House Prices')
plt.show()
