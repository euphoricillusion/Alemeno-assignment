import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have already computed these values:
similarity_alphabet_tesla = 0.72
similarity_alphabet_uber = 0.97
similarity_tesla_uber = 0.70

# Store the similarities in a list
similarity_scores = [similarity_alphabet_tesla, similarity_alphabet_uber, similarity_tesla_uber]
comparisons = ['Alphabet vs Tesla', 'Alphabet vs Uber', 'Tesla vs Uber']

# Heatmap Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(
    [[1.0, similarity_alphabet_tesla, similarity_alphabet_uber],
     [similarity_alphabet_tesla, 1.0, similarity_tesla_uber],
     [similarity_alphabet_uber, similarity_tesla_uber, 1.0]],
    annot=True, cmap='coolwarm', xticklabels=['Alphabet', 'Tesla', 'Uber'], 
    yticklabels=['Alphabet', 'Tesla', 'Uber'], fmt='.2f'
)
plt.title("Cosine Similarity Heatmap")
plt.show()  # Display the heatmap

# **Add a pause before the next plot**
plt.close()  # Close the previous plot to ensure no overlap

# Bar Chart Visualization
plt.figure(figsize=(8, 5))
plt.bar(comparisons, similarity_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
plt.ylim(0, 1)
plt.title("Cosine Similarity between Documents")
plt.ylabel("Cosine Similarity")
plt.show()  # Display the bar chart
