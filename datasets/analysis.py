from collections import Counter
import matplotlib.pyplot as plt

def analyze_usure_dataset(dataset):
    """Affiche la distribution des classes dans le dataset."""
    class_counts = Counter()
    for _, target in dataset:
        for l in target['labels'].tolist():
            class_counts[l] += 1
    idx_to_class = {v: k for k, v in dataset.cls_to_idx.items()}
    print("\n=== Distribution des classes ===")
    for idx, count in sorted(class_counts.items()):
        cname = idx_to_class.get(idx, f"cls_{idx}")
        print(f"{cname:<15}: {count:5d}")
    total = sum(class_counts.values())
    print(f"\nTotal objets annotés: {total}")
    return class_counts


def plot_class_distribution(class_counts, dataset):
    """Trace un histogramme des classes."""
    idx_to_class = {v: k for k, v in dataset.cls_to_idx.items()}
    labels = [idx_to_class[i] for i in class_counts.keys()]
    values = [class_counts[i] for i in class_counts.keys()]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Distribution des classes dans le dataset")
    plt.tight_layout()
    plt.show()

