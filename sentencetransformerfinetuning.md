
## Fine-Tuning Script (using sentence-transformers)

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

# Step 1: Prepare your training examples
train_examples = [
    InputExample(texts=["Mobile Automation with Appium", "Appium mobile testing"], label=0.9),
    InputExample(texts=["API Testing", "REST API automation"], label=0.85),
    InputExample(texts=["Java 17", "Java programming"], label=0.95),
    InputExample(texts=["DevOps", "Frontend Development"], label=0.1),
    InputExample(texts=["Manual testing", "Selenium"], label=0.3),
    InputExample(texts=["No Skill", "No Skill"], label=1.0),
    # Add many more examples to improve quality
]

# Step 2: Load a base model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Step 3: Create a DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# Step 4: Define a loss function
train_loss = losses.CosineSimilarityLoss(model)

# Step 5: Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,                    # Increase to 3–5 for better training
    warmup_steps=10,
    show_progress_bar=True
)

# Step 6: Save your custom model
model_save_path = "output/skill_match_model"
model.save(model_save_path)

print(f"Model saved at {model_save_path}")

```


## After Training: How to Use the Model

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("output/skill_match_model")

# Example skill matching
s1 = "Mobile Automation with Appium"
s2 = "Appium mobile testing"

embedding1 = model.encode(s1, convert_to_tensor=True)
embedding2 = model.encode(s2, convert_to_tensor=True)

similarity = util.cos_sim(embedding1, embedding2).item()
print(f"Similarity: {similarity:.2f}")

```

## What the model does (before fine-tuning)

- By default, the model has been trained on general sentence pairs like:

- "How can I reset my password?"

- "I forgot my login info"
- Label: 0.9 (meaning "similar")

It learned to generate embeddings that make similar sentences closer together using cosine similarity.


## So why give InputExample(texts=["A", "B"], label=0.9)

It trains the model using a loss function (like cosine similarity loss or triplet loss) to adjust its internal weights so that:

```python
cosine(embedding(A), embedding(B)) ≈ label

```

This means:
- The embeddings for similar skills will be closer together
- The embeddings for unrelated skills will be further apart


## Why fine-tune?

Out of the box, sentence-transformer models are trained on generic text pairs (like FAQ, Quora, paraphrases).

But in this use case (e.g., matching candidate skills with job requirements) is more specific.

Fine-tuning helps the model learn that:

- "Appium testing" and "Mobile Automation with Appium" are highly similar
- "DevOps" and "Manual Testing" are not similar — even if they’re both "tech skills"


you're fine-tuning the model to produce embeddings that make cosine similarity more meaningful.

