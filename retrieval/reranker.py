from sentence_transformers import CrossEncoder


print("=" * 60)
print("Loading BGE Reranker...")
print("=" * 60)

model = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Model loaded successfully!\n")


query = "What is cyber security?"


documents = [

    "Cyber security protects systems from digital attacks.",

    "Nutrition is the science of food and health.",

    "Food preservation increases shelf life.",

    "Cyber security includes network security, data protection and risk management."
]


pairs = []

for doc in documents:

    pairs.append(
        [query, doc]
    )


scores = model.predict(pairs)


print("=" * 60)
print("RERANKING RESULTS")
print("=" * 60)

for doc, score in zip(documents, scores):

    print("\nDocument:")
    print(doc)

    print(f"\nScore: {score:.4f}")

    print("-" * 60)