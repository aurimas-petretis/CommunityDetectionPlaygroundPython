import csv
from collections import defaultdict

# Input file path
input_filename = 'lfr_benchmark_with_must_link_results.csv'

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

# Fields to average (exclude Must-link edges and Index)
fields_to_average = [field for field in header if field not in ['Must-link edges', 'Index']]

# Compute and print averages
print("Average values per Must-link edges:\n")

for must_link in sorted(grouped_by_must_link, key=lambda x: int(x)):
    rows = grouped_by_must_link[must_link]
    num_rows = len(rows)
    print(f"Must-link edges: {must_link}")

    for field in fields_to_average:
        total = sum(float(row[field]) for row in rows)
        avg = total / num_rows
        print(f"  {field}: {avg:.4f}")
    print()
