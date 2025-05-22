# filename = 'lfr_benchmark_with_must_link_averaged_results.csv'
#
# def plot_xy():
#     import datetime as dt
#     import matplotlib.pyplot as plt
#
#     x = [dt.datetime.strptime(d,'%d.%m.%Y').date() for d in dates.values[:-1]]
#     y = returns
#
#     plt.figure(figsize=(12, 6))
#     plt.title('Baltic Benchmark Index Returns (2005-05-01 - 2025-05-01)', fontsize=14)
#     plt.xlabel('Date')
#     plt.ylabel('Returns')
#     plt.plot(x,y)
#     plt.gcf().autofmt_xdate()
#     plt.grid(True)
#     plt.savefig('lfr_benchmark_with_must_link_averaged_results.png', dpi=300, bbox_inches='tight')
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("lfr_benchmark_with_must_link_averaged_results.csv")

# Clean column names (in case of extra spaces)
df.columns = df.columns.str.strip()

# Convert columns to appropriate types
df['Must-link edges'] = pd.to_numeric(df['Must-link edges'], errors='coerce')
df['Modularity'] = pd.to_numeric(df['Modularity'], errors='coerce')

# Sort by Must-link edges
df_sorted = df.sort_values(by='Must-link edges')

# Plot using Seaborn for better aesthetics
plt.figure(figsize=(10, 6))
sns.lineplot(x='Must-link edges', y='Modularity', data=df_sorted, marker='o')

plt.title('Modularity vs Must-link Constraints')
plt.xlabel('Must-link edges')
plt.ylabel('Modularity')
plt.grid(True)
plt.tight_layout()
plt.savefig('lfr_benchmark_with_must_link_averaged_results.png', dpi=300, bbox_inches='tight')
plt.show()
