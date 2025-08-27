import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction dataset
dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Bread'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter']
]


all_items = sorted(set(item for transaction in dataset for item in transaction))
one_hot = []
for transaction in dataset:
    one_hot.append([1 if item in transaction else 0 for item in all_items])
df = pd.DataFrame(one_hot, columns=all_items)

# Apply Apriori (minimum support 0.6)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
print("Frequent Itemsets:\n", frequent_itemsets)

# Generate association rules (min confidence 0.7)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:\n", rules)