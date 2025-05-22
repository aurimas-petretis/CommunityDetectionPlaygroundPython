import csv
from collections import defaultdict

import pandas as pd

# Input and output file names
input_filename = 'lfr_benchmark_with_must_link_results.csv'
output_filename = 'lfr_benchmark_with_must_link_averaged_results_ml0.csv'

# Grouped data: Must-link edges -> list of rows
grouped_by_must_link = defaultdict(list)

# Read and clean the CSV
with open(input_filename, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    reader.fieldnames = [h.strip() for h in reader.fieldnames]
    header = reader.fieldnames

    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items()}
        grouped_by_must_link[row['Must-link edges']].append(row)

# Fields to average (excluding Must-link edges and Index)
fields_to_average = [field for field in header if field not in ['Must-link edges', 'Index']]

# Prepare average results
average_results = []

for must_link in sorted(grouped_by_must_link, key=lambda x: int(x)):
    rows = grouped_by_must_link[must_link]
    num_rows = len(rows)
    avg_row = {'Must-link edges': must_link}

    for field in fields_to_average:
        total = sum(float(row[field]) for row in rows)
        avg = total / num_rows
        avg_row[field] = round(avg, 6)

    average_results.append(avg_row)

# Write to output CSV
output_fields = ['Must-link edges'] + fields_to_average

with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=output_fields)
    writer.writeheader()
    for row in average_results:
        writer.writerow(row)

with open("lfr_benchmark_with_must_link_averaged_results.tex", 'w') as f:
    csv_table = pd.read_csv("lfr_benchmark_with_must_link_averaged_results.csv")
    f.write(csv_table.to_latex(index=False))

print(f"Averages per 'Must-link edges' have been written to '{output_filename}'.")
