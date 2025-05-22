import csv
from collections import defaultdict

# File names
input_filename = 'lfr_benchmark_with_must_link_results.csv'
output_filename = 'lfr_benchmark_with_must_link_results_grouped.csv'

# Dictionary to hold rows grouped by Index
grouped_data = defaultdict(list)

# Read the CSV and group by Index
with open(input_filename, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    header = reader.fieldnames

    for row in reader:
        index = row['Index']
        grouped_data[index].append(row)

# Write the grouped data to output CSV
with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()

    for index in sorted(grouped_data, key=lambda x: int(x)):  # Sort by numeric Index
        for row in grouped_data[index]:
            writer.writerow(row)

print(f"Data grouped by 'Index' has been written to '{output_filename}'.")
