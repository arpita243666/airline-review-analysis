
# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, chi2_contingency, shapiro

# Global style
sns.set(style="whitegrid", palette="deep")

# 2. Load Dataset

df = pd.read_csv("cleaned_airline_review.csv")
print("Dataset Loaded!\n")


# 3. Basic Info

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())

print("\nInfo:\n")
df.info()

print("\nMissing Values:\n")
print(df.isnull().sum())


# 4. Data Cleaning

df.drop(columns=[col for col in ['Unnamed: 0'] if col in df.columns], inplace=True)

df['overall_rating'] = pd.to_numeric(df['overall_rating'], errors='coerce')

df.dropna(inplace=True)

print("\nCleaned Data Shape:", df.shape)


# 5. Synthetic Location Data

np.random.seed(42)
df['latitude'] = np.random.uniform(-90, 90, len(df))
df['longitude'] = np.random.uniform(-180, 180, len(df))


# 6. EDA

print("\nSummary Statistics:\n", df.describe())

avg_rating = df.groupby('airline_name')['overall_rating'] \
               .mean().sort_values(ascending=False)

print("\nTop Airline Ratings:\n", avg_rating.head(10))


# 7. Visualizations


# 1. Geo Scatter Plot
plt.figure(figsize=(10,6))
scatter = plt.scatter(
    df['latitude'],
    df['longitude'],
    c=df['overall_rating'],
    cmap='plasma',
    alpha=0.7
)
plt.colorbar(scatter, label='Rating')
plt.title("Airline Review Distribution (Synthetic Geo View)")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

# 2. Top 10 Airlines by Reviews
top_airlines = df['airline_name'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(
    x=top_airlines.index,
    y=top_airlines.values,
    hue=top_airlines.index,        
    palette='viridis',
    legend=False                  
)
plt.title("Top 10 Airlines by Number of Reviews")
plt.xticks(rotation=45)
plt.xlabel("Airline")
plt.ylabel("Review Count")
plt.show()

# 3. Rating Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['overall_rating'], kde=True, color='skyblue')
plt.title("Rating Distribution")
plt.show()

# ================================
# Normal Distribution Overlay
# ================================
mean = df['overall_rating'].mean()
std = df['overall_rating'].std()

plt.figure(figsize=(8,5))

sns.histplot(
    df['overall_rating'],
    kde=False,
    stat='density',
    bins=20,
    color='skyblue',
    label='Data'
)

x = np.linspace(df['overall_rating'].min(), df['overall_rating'].max(), 100)
y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean)/std)**2)

plt.plot(x, y, color='red', linewidth=2, label='Normal Distribution')

plt.title("Normal Distribution Fit")
plt.xlabel("Rating")
plt.ylabel("Density")
plt.legend()
plt.show()

print(f"\nMean Rating: {mean:.2f}, Std Dev: {std:.2f}")

# 4. Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=df['overall_rating'], color='orange')
plt.title("Outlier Detection (Ratings)")
plt.show()

# 5. Top 10 Airlines by Average Rating
top_avg_rating = df.groupby('airline_name')['overall_rating'] \
                   .mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(
    x=top_avg_rating.index,
    y=top_avg_rating.values,
    hue=top_avg_rating.index,     
    palette='coolwarm',
    legend=False                 
)
plt.title("Top 10 Airlines by Average Rating")
plt.xticks(rotation=45)
plt.xlabel("Airline")
plt.ylabel("Average Rating")
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(
    df.select_dtypes(include=['number']).corr(),
    annot=True,
    cmap='RdYlBu'
)
plt.title("Correlation Heatmap")
plt.show()


# 8. Outlier Detection

Q1 = df['overall_rating'].quantile(0.25)
Q3 = df['overall_rating'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[
    (df['overall_rating'] < Q1 - 1.5 * IQR) |
    (df['overall_rating'] > Q3 + 1.5 * IQR)
]

print("\nNumber of Outliers:", len(outliers))


# 9. Statistical Analysis


# Normality Test
sample_data = df['overall_rating'].sample(min(500, len(df)))
stat, p = shapiro(sample_data)

print("\nShapiro Test p-value:", p)

if p > 0.05:
    print("Data is Normally Distributed")
else:
    print("Data is NOT Normally Distributed")

# T-test
airlines = df['airline_name'].unique()

if len(airlines) >= 2:
    a1 = df[df['airline_name'] == airlines[0]]['overall_rating']
    a2 = df[df['airline_name'] == airlines[1]]['overall_rating']

    stat, p = ttest_ind(a1, a2, equal_var=False)
    print(f"\nT-test between {airlines[0]} and {airlines[1]}:")
    print("p-value:", p)

# Chi-Square
if 'recommended' in df.columns:
    contingency = pd.crosstab(df['airline_name'], df['recommended'])
    stat, p, dof, expected = chi2_contingency(contingency)

    print("\nChi-Square Test p-value:", p)


# 10. Conclusion

print("\nProject Completed Successfully!")
