# Named Entity Recognition (NER)

## What Is It?

NER finds and labels **specific things** in text — names of people, companies, locations, dates, amounts, and more. It extracts structured data from unstructured text.

```
Input:  "Apple CEO Tim Cook announced iPhone 16 in Cupertino on September 9."

Output:  [Apple]ORG CEO [Tim Cook]PERSON announced [iPhone 16]PRODUCT
         in [Cupertino]LOCATION on [September 9]DATE
```

## Frontend Analogy

```javascript
// NER is like regex extraction, but it understands MEANING:

// Regex approach (brittle):
const dates = text.match(/\d{1,2}\/\d{1,2}\/\d{4}/g);   // only matches "12/25/2024"
// Misses: "December 25th", "Christmas Day", "next Tuesday"

// NER approach (understands meaning):
ner("Meet me next Tuesday at the Apple Store in NYC")
// → { dates: ["next Tuesday"], orgs: ["Apple"], locations: ["NYC"] }
// It understands "next Tuesday" IS a date even without numbers

// Think of NER as intelligent form auto-fill:
// User types: "Send $500 to John Smith at Bank of America by Friday"
// NER extracts: { amount: "$500", person: "John Smith",
//                 org: "Bank of America", date: "Friday" }
```

## Standard Entity Types

| Label | Meaning | Examples |
|-------|---------|---------|
| **PER / PERSON** | People | Tim Cook, Marie Curie |
| **ORG** | Organizations | Apple, United Nations, MIT |
| **LOC / GPE** | Locations | New York, France, Mount Everest |
| **DATE** | Dates & times | September 9, 2024, next week |
| **MONEY** | Monetary values | $500, 10 million euros |
| **PRODUCT** | Products | iPhone 16, Tesla Model 3 |
| **EVENT** | Events | World Cup, WWDC |
| **MISC** | Other entities | English (language), Python (lang) |

## Real-World Use Cases

- **Chatbots** — extract user intent: "Book a flight to **Paris** for **2 people** on **March 5th**"
- **Search** — understand queries: "**Tesla** stock price **today**"
- **Content moderation** — detect PII: redact names, emails, phone numbers
- **Knowledge graphs** — build entity relationships from news articles
- **Healthcare** — extract drug names, symptoms, conditions from medical notes
- **Resume parsing** — extract name, skills, companies, education

## How NER Works

### Token-Level Classification

NER is classification **per token**, not per sentence:

```
Text:    "Tim   Cook   works   at   Apple"
Labels:  B-PER  I-PER  O       O    B-ORG

B- = Beginning of entity
I- = Inside entity (continuation)
O  = Outside (not an entity)
```

This is called **BIO tagging** (Begin, Inside, Outside):

```
"New York City is great"
 B-LOC I-LOC I-LOC O  O

"New" = beginning of location
"York" = inside (continuation of) location
"City" = still inside location
"is" = outside (not an entity)
"great" = outside
```

### Why BIO Tagging?

Without it, you can't tell where one entity ends and another starts:

```
"Tim Cook met Jony Ive"

Without BIO:  PER PER O PER PER    → Is it "Tim Cook met Jony Ive" (2 people)?
                                       Or "Tim Cook met Jony" + "Ive" (3 entities)?

With BIO:     B-PER I-PER O B-PER I-PER → Clear: 2 people
```

## Three Ways to Do NER

### 1. Pretrained Pipeline (Fastest Start)

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)

text = "Elon Musk founded SpaceX in 2002 in Los Angeles"
entities = ner(text)

for entity in entities:
    print(f"  {entity['word']:20} → {entity['entity_group']:8} ({entity['score']:.0%})")

# Output:
#   Elon Musk            → PER      (99%)
#   SpaceX               → ORG      (98%)
#   2002                 → DATE     (95%)
#   Los Angeles          → LOC      (99%)
```

### 2. spaCy (Best for Production Pipelines)

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(f"  {ent.text:20} → {ent.label_:8} ({ent.start_char}-{ent.end_char})")

# Output:
#   Apple                → ORG      (0-5)
#   U.K.                 → GPE      (27-31)
#   $1 billion           → MONEY    (44-54)
```

### 3. LLM Prompting (Most Flexible)

```python
prompt = """Extract all named entities from this text.
Return JSON with keys: persons, organizations, locations, dates.

Text: "Tim Cook announced that Apple will open a new office in Tokyo by March 2025."

JSON:"""

# LLM responds:
# {
#   "persons": ["Tim Cook"],
#   "organizations": ["Apple"],
#   "locations": ["Tokyo"],
#   "dates": ["March 2025"]
# }
```

**When to use:** Custom entity types, no training data, complex extraction. More expensive but infinitely flexible.

## Comparison

| | Pretrained Pipeline | spaCy | LLM Prompting |
|---|---|---|---|
| **Setup** | 2 lines of code | Install + load model | API key |
| **Speed** | Fast | Very fast | Slow |
| **Custom entities** | Needs fine-tuning | Trainable | Just describe them |
| **Accuracy** | High for standard entities | High | Good to great |
| **Cost** | Free | Free | $ per call |
| **Best for** | Quick extraction | Production pipelines | Custom/complex entities |

## Fine-Tuning NER for Custom Entities

When you need to detect **domain-specific entities** (product SKUs, medical terms, legal clauses):

```python
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    Trainer, TrainingArguments
)
from datasets import load_dataset

# Load a NER dataset (or prepare your own)
dataset = load_dataset("conll2003")

# Load pretrained model
model_name = "bert-base-cased"  # cased matters for NER!
tokenizer = AutoTokenizer.from_pretrained(model_name)

label_list = dataset["train"].features["ner_tags"].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
)

# Tokenize (NER needs special handling — align labels with subword tokens)
def tokenize_and_align(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)       # ignore special tokens
            elif word_id != prev_word:
                label_ids.append(label[word_id])  # first token of word
            else:
                label_ids.append(-100)       # ignore subword continuations
            prev_word = word_id
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align, batched=True)

# Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./ner-results",
        num_train_epochs=3,
        learning_rate=2e-5,
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)
trainer.train()
```

## Evaluation Metrics for NER

NER uses **entity-level** F1, not token-level:

```
Text:     "Tim Cook works at Apple"
Predicted: [Tim Cook]PER works at [Apple Inc]ORG
Actual:    [Tim Cook]PER works at [Apple]ORG

"Tim Cook" → exact match ✓
"Apple Inc" vs "Apple" → partial match ✗ (strict) or ✓ (lenient)
```

```python
from seqeval.metrics import classification_report

# seqeval is the standard NER evaluation library
print(classification_report(y_true_tags, y_pred_tags))
#               precision  recall  f1-score  support
#   LOC           0.93     0.91     0.92     1668
#   ORG           0.88     0.85     0.86     1661
#   PER           0.96     0.95     0.95     1617
#   micro avg     0.92     0.90     0.91     4946
```

## Key Takeaway

NER extracts **structured information from unstructured text** — people, places, companies, dates, and any custom entity type you define. For standard entities, use `pipeline("ner")` or spaCy — they work out of the box. For custom entities, either fine-tune a BERT model or prompt an LLM. NER is essential for chatbots (intent extraction), search systems, and data pipelines where you need to turn free text into structured data.
