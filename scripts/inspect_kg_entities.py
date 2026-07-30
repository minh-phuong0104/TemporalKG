import json
from collections import Counter

from scripts.config import GRAPH_JSON_FILE


def main():

    graph = json.loads(
        GRAPH_JSON_FILE.read_text(encoding="utf-8")
    )

    nodes = graph["nodes"]

    print("=" * 60)
    print("KG ENTITY INSPECTION")
    print("=" * 60)

    print(f"Total nodes: {len(nodes)}")

    labels = []

    for node in nodes:
        label = node.get("label", node["id"])
        labels.append(label)


    print("\nSample entities:")
    print("-" * 60)

    for e in labels[:100]:
        print(e)


    print("\nLong entities (>6 words)")
    print("-" * 60)

    count = 0

    for e in labels:

        if len(e.split()) > 6:
            print(e)
            count += 1

            if count >= 100:
                break


    print("\nStatistics")
    print("-" * 60)

    lengths = Counter(
        len(e.split())
        for e in labels
    )

    print(lengths)


if __name__ == "__main__":
    main()