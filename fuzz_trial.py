from fuzzywuzzy import process
import re
import pandas as pd

# Load the dataset
df = pd.read_csv('Datasets/DataColl.csv', encoding='latin-1')

# Drop the first row if needed (assuming it's a header row)
df = df.drop(0)

# Convert the 'Section' column to a list
sections = df['Section'].to_list()

# Choose the closest in terms of characters
query = '610/229/125 UB'
ratios = process.extract(query, sections, limit=len(sections))

# Sort ratios by score in descending order to get the highest at the top
ratios.sort(key=lambda x: x[1], reverse=True)

# Find the highest score
max_best_ratio = ratios[0][1]
print(f"Max Best Ratio: {max_best_ratio}")

# Filter the highest ratios
filtered_ratios = []
i = 0
while i < len(ratios) and ratios[i][1] == max_best_ratio:
    filtered_ratios.append(ratios[i][0])
    i += 1

print(f"Filtered Ratios: {filtered_ratios}")

# Extracting the numbers from the original query
numbers = re.findall(r'[/_xX]*\d+', query)
extracted_numbers = ''.join(numbers)
parts = re.split('/|_|x|X', extracted_numbers)

# Remove empty strings from parts
parts = [part for part in parts if part]

print(f"Extracted Parts from Query: {parts}")

# Extracting the numbers from the highest ratio matches
for ratio in filtered_ratios:
    numbers_in_ratio = re.findall(r'[/_\dX ]*\d+', ratio)
    extracted_numbers_in_ratio = ''.join(numbers_in_ratio)
    parts_in_ratio = re.split('/|_|x|X', extracted_numbers_in_ratio)
    parts_in_ratio = [part for part in parts_in_ratio if part]
    print(f"Parts from {ratio}: {parts_in_ratio}")