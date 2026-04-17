# Named Entity Recognition (NER)

## 1. TL;DR

NER finds and labels **specific named things** in text — people, organizations, locations, dates, money amounts. It turns unstructured text into structured data. Use `pipeline("ner", grouped_entities=True)` or spaCy for standard entities out of the box. Use LLM prompting for custom entity types without training. Fine-tune BERT (`AutoModelForTokenClassification`) when you need custom entities at high volume. Key difference from text classification: NER classifies each **token**, not the whole sentence.

---

## 2. The Mental Model

> 💡 **Think of it like this:** NER is an **intelligent highlighter** that color-codes text by the type of thing it refers to.

A legal assistant reviewing a contract highlights all dates in yellow, all company names in blue, all dollar amounts in green. NER does this automatically — reading the entire sentence context to decide what color each word gets.

| Real world | Technical concept |
|---|---|
| Color-coding important terms in a contract | Assigning entity labels to tokens |
| Yellow = date, blue = company, green = money | DATE, ORG, MONEY entity types |
| "Apple" → blue (context: tech company) | Context-dependent labeling |
| Starting vs. continuing a highlighted phrase | B- (Begin) vs. I- (Inside) tags |
| Reviewing the whole sentence before highlighting | Bidirectional attention in BERT |

---

## 3. Why It Exists

**The problem:** Enormous volumes of text — news articles, medical records, legal documents, support tickets — contain valuable structured information trapped in unstructured sentences. "Tim Cook announced Apple will open a Tokyo office by March 2025" contains 4 pieces of structured data. Extracting them manually at scale is impossible.

**What came before:** Rule-based extraction — regex patterns like `\d{1,2}/\d{1,2}/\d{4}` for dates. This fails immediately on "next Tuesday," "the first quarter," "Christmas Day," or any other natural language date expression.

**What changed:** Statistical models (CRFs) learned to recognize entities by their context — words around them, capitalization patterns, position in the sentence. Transformer models (BERT) pushed accuracy to near-human levels by deeply understanding context. Now "Apple" gets tagged ORG when surrounded by business language and stays untagged when discussing apple pie.

---

## 4. Core Concepts

### BIO Tagging

**One-line definition:** A labeling scheme where B=Begin entity, I=Inside entity, O=Outside (not an entity).

**Analogy:** Like using parentheses to group words. The opening `(` is B-, everything inside is I-, and everything outside is O. Without B/I distinction, you can't tell where one entity ends and another starts.

```
"Tim Cook met Jony Ive in California"

Without BIO:  PER PER O  PER PER  O  LOC
              → Is it one person "Tim Cook Jony Ive"? Three people?

With BIO:     B-PER I-PER O  B-PER I-PER O  B-LOC
              → Clear: two people (Tim Cook, Jony Ive) + one location
```

The full tag set for a model with 4 entity types:
```
O       = not an entity
B-PER   = beginning of a person
I-PER   = inside a person entity
B-ORG   = beginning of an organization
I-ORG   = inside an organization
B-LOC   = beginning of a location
I-LOC   = inside a location
B-MISC  = beginning of miscellaneous entity
I-MISC  = inside miscellaneous entity
```

**Common misconception:** ❌ "Each word gets one label" → ✅ Each subword *token* gets a label. "New York" is two words but might tokenize into `["New", "York"]` or `["New", "Y", "##ork"]`. NER pipelines merge subword tokens back into word-level entities.

> 💡 **Why this matters for fine-tuning:** Your labels are at the word level (`"New York" → B-LOC I-LOC`), but the model sees subword tokens. You must align them — typically: label the **first** subword of each word with the real tag, and set continuation subwords to `-100` (a special value that PyTorch's cross-entropy loss ignores). Search for "word_ids() NER alignment" in the HF docs — that's the canonical pattern.

---

### Token-Level Classification

**One-line definition:** Unlike sentence classification (one label per sentence), NER assigns a label to every token in the sequence.

**Analogy:** Text classification = grading an essay with one letter grade (A/B/C). NER = annotating every sentence in the essay with a specific comment.

```
Input:         "Apple CEO Tim Cook announced iPhone 16"
Tokens:         Apple  CEO  Tim   Cook  announced  iPhone  16
NER labels:     B-ORG   O  B-PER I-PER     O       B-PROD  I-PROD
```

The model architecture is the same as BERT classification, except instead of using the [CLS] vector, you use **every token's output vector** to make a prediction.

**Common misconception:** ❌ "I need a separate model for each entity type" → ✅ One model predicts all entity types simultaneously — it outputs a label distribution over [O, B-PER, I-PER, B-ORG, ...] for each token in one forward pass.

---

### Standard Entity Types

**One-line definition:** A shared vocabulary of entity categories used across most NLP datasets and models.

| Label | Meaning | Examples |
|---|---|---|
| **PER** | People | Tim Cook, Marie Curie |
| **ORG** | Organizations | Apple, United Nations |
| **LOC / GPE** | Locations / geopolitical | New York, France |
| **DATE** | Dates and times | September 9, next week |
| **MONEY** | Monetary values | $500, 10 million euros |
| **PRODUCT** | Products | iPhone 16, Tesla Model 3 |
| **EVENT** | Named events | World Cup, WWDC |
| **MISC** | Other notable entities | Python (language) |

**Common misconception:** ❌ "All models use the same labels" → ✅ CoNLL-2003 (standard benchmark) uses PER/ORG/LOC/MISC. spaCy uses different labels (PERSON, ORG, GPE, DATE...). OntoNotes uses 18 types. Always check your model's label set.

---

### Grouped Entities

**One-line definition:** Merging consecutive tokens with the same entity type into a single entity span.

**Analogy:** Autocomplete merges "N" + "e" + "w" + " " + "Y" + "o" + "r" + "k" into a single suggestion "New York." Grouping entities does the same for BIO-tagged token sequences.

```python
from transformers import pipeline

ner = pipeline("ner")  # without grouping
result = ner("Tim Cook founded Apple")
# [{'entity': 'B-PER', 'word': 'Tim', ...},
#  {'entity': 'I-PER', 'word': 'Cook', ...},
#  {'entity': 'B-ORG', 'word': 'Apple', ...}]

ner_grouped = pipeline("ner", grouped_entities=True)  # with grouping
result = ner_grouped("Tim Cook founded Apple")
# [{'entity_group': 'PER', 'word': 'Tim Cook', 'score': 0.99},
#  {'entity_group': 'ORG', 'word': 'Apple', 'score': 0.98}]
```

Always use `grouped_entities=True` in production — individual subword tokens are rarely what you want.

---

## 5. How It Actually Works (Step-by-Step)

Let's trace "Elon Musk founded SpaceX in 2002 in Los Angeles" through a BERT NER model:

```
INPUT: "Elon Musk founded SpaceX in 2002 in Los Angeles"

Step 1: Tokenize
  ["[CLS]", "Elon", "Musk", "founded", "Space", "##X", "in", "2002",
   "in", "Los", "Angeles", "[SEP]"]

Step 2: Embed + positional encoding
  Each token → 768-dimensional vector

Step 3: Pass through 12 BERT encoder layers (bidirectional)
  "Musk" attends to "Elon" and "founded" → learns: human name
  "SpaceX" attends to "founded" and "Musk" → learns: founded company
  "2002" attends to "in", "founded" → learns: date context

Step 4: For EACH token, run classification head
  Token:    Elon    Musk   founded  SpaceX  in   2002   in   Los    Angeles
  Labels:  B-PER  I-PER    O      B-ORG   O   B-DATE  O   B-LOC  I-LOC

Step 5: Group consecutive same-type tokens
  (Elon, Musk) → PER entity: "Elon Musk"
  (SpaceX) → ORG entity: "SpaceX"
  (2002) → DATE entity: "2002"
  (Los, Angeles) → LOC entity: "Los Angeles"

OUTPUT:
  PER:  "Elon Musk"    (99%)
  ORG:  "SpaceX"       (98%)
  DATE: "2002"         (95%)
  LOC:  "Los Angeles"  (99%)
```

> 💡 **Key Insight:** BERT's bidirectional attention means "SpaceX" can see both "founded" (before it) and "in 2002" (after it). This context confirms it's an organization, not a random word. A left-to-right model would only have "founded" as context when processing "SpaceX."

---

## 6. Code in Practice

### Minimal: Out-of-the-box NER

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)

text = "Apple CEO Tim Cook announced iPhone 16 in Cupertino on September 9."
entities = ner(text)

for entity in entities:
    print(f"  {entity['word']:20} → {entity['entity_group']:8} ({entity['score']:.0%})")

# Apple                → ORG      (99%)
# Tim Cook             → PER      (99%)
# iPhone 16            → MISC     (94%)
# Cupertino            → LOC      (99%)
# September 9          → DATE     (95%)
```

### Practical: spaCy for production pipelines

```python
import spacy

nlp = spacy.load("en_core_web_sm")  # pip install spacy && python -m spacy download en_core_web_sm

doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(f"  {ent.text:20} → {ent.label_:8}")

# Apple                → ORG
# U.K.                 → GPE
# $1 billion           → MONEY

# Also access positions:
print(f"Start: {ent.start_char}, End: {ent.end_char}")
```

### Real-world pattern: LLM-based extraction for custom entities

```python
import anthropic
import json

client = anthropic.Anthropic()

def extract_entities(text: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Extract named entities from the text below.
Return JSON with keys: persons, organizations, locations, dates, amounts.
Only include entities that are explicitly mentioned.

Text: "{text}"

JSON:"""
        }]
    )
    return json.loads(response.content[0].text)

result = extract_entities(
    "Tim Cook announced Apple will open a new office in Tokyo by March 2025."
)
print(result)
# {
#   "persons": ["Tim Cook"],
#   "organizations": ["Apple"],
#   "locations": ["Tokyo"],
#   "dates": ["March 2025"],
#   "amounts": []
# }
```

---

## 7. Gotchas & Pitfalls

❌ **Not using `grouped_entities=True`** → ✅ Without it, you get individual subword tokens labeled separately ("New" B-LOC, "York" I-LOC, "##shire" I-LOC). Always group for usable output.

❌ **Using bert-base-uncased for NER** → ✅ Capitalization is a major signal for proper nouns. Always use `bert-base-cased` for NER tasks — "Apple" (company) vs. "apple" (fruit) is only distinguishable by the capital A.

❌ **Ignoring entity boundaries in evaluation** → ✅ Partial matches count as wrong in strict evaluation. "Apple Inc" vs. "Apple" are different spans even if both are ORG. Use `seqeval` for proper entity-level F1.

❌ **Assuming standard entities fit your domain** → ✅ General NER models don't know about your domain's entities. A medical NER needs separate training on clinical text; a legal NER needs legal documents. Domain shift is real.

❌ **Subword alignment bugs in fine-tuning** → ✅ When tokenizing for NER fine-tuning, subword tokens from the same word must share the same label (or use -100 to ignore continuation tokens). This is the trickiest part of NER fine-tuning — use the `is_split_into_words=True` + `word_ids()` pattern.

❌ **Treating overlapping entities as impossible** → ✅ Standard BIO can't represent "New York City" as both LOC and part of "New York City Marathon" (EVENT). For overlapping entities, use different annotation schemes or post-process.

---

## 8. When to Use / When NOT to Use

### Use NER when:
- **Information extraction** — pull structured fields from free text (invoices, resumes, medical notes)
- **Knowledge graph building** — find all entities and their relationships
- **PII detection/redaction** — find names, emails, phone numbers in documents
- **Search enrichment** — index entities for faceted filtering ("all articles mentioning Apple")
- **Chatbot intent parsing** — extract "Paris" from "book a flight to Paris for March 5"

### Don't use NER when:
- **Regex is sufficient** — structured formats like emails, URLs, phone numbers in a consistent format
- **You only care about a few specific strings** — exact string matching is faster and more precise
- **Real-time with <10ms latency** — BERT inference adds latency; consider spaCy's faster models or CRF-based approaches
- **General information extraction** — if your "entities" are really relationships or complex facts, use LLM prompting or information extraction frameworks

---

## 9. Related Concepts (The Map)

- **Text Classification** — NER is the token-level version of classification. Same BERT backbone, different head: `AutoModelForTokenClassification` instead of `AutoModelForSequenceClassification`.
- **BERT** — the engine behind most NER systems. BERT's bidirectional attention gives it the context awareness needed to distinguish "Apple" (company) from "apple" (fruit).
- **Relation Extraction** — the step after NER: given entities, find the relationships between them ("Tim Cook" [CEO_OF] "Apple"). Harder and less standardized than NER.
- **Information Extraction (IE)** — the broader field NER belongs to. IE includes NER + relation extraction + event extraction + coreference resolution.
- **spaCy** — the production-grade NLP library. Built on top of modern Transformer models but wrapped in a fast, production-friendly pipeline. The go-to for NER at scale.

---

## 10. Cheat Sheet

| Approach | Accuracy | Speed | Custom Entities | Cost |
|---|---|---|---|---|
| `pipeline("ner")` | High | Fast | No (standard only) | Free |
| spaCy `en_core_web_sm` | High | Very fast | Trainable | Free |
| LLM prompting | Good | Slow | Yes (just describe) | $ per call |
| Fine-tuned BERT | Best | Fast | Yes | GPU training |

**BIO tag reference:**
```
O       = not an entity (most tokens)
B-TYPE  = first token of an entity
I-TYPE  = continuation token within an entity
```

**Core usage pattern:**
```python
ner = pipeline("ner", grouped_entities=True)
entities = ner("your text here")
# [{'entity_group': 'PER', 'word': '...', 'score': 0.99, 'start': N, 'end': N}]
```

**Remember this:**
1. NER = per-token classification (not per-sentence)
2. Always `grouped_entities=True` and always use cased models
3. For custom entities with no training data → LLM prompting

---

## 11. Self-Check Questions

1. What problem does BIO tagging solve that simple entity-type labeling can't?
2. Why should you use `bert-base-cased` instead of `bert-base-uncased` for NER?
3. "I got the spaCy model working on English news. Now I need it to work on medical records." What should you expect?
4. What's the difference between NER's evaluation metric and text classification's?
5. When would you choose LLM-based extraction over a fine-tuned BERT NER model?

<details>
<summary>Answers</summary>

1. Without BIO tagging, you can't tell where one entity ends and another begins. If tokens are labeled [PER, PER, O, PER, PER], you can't distinguish one 4-word person from two 2-word people. B- marks the start of a new entity, I- marks continuations. With BIO, [B-PER, I-PER, O, B-PER, I-PER] unambiguously represents two separate people.

2. Capitalization is a critical signal in NER. "Apple" (company), "Apple Store" (location), vs. "apple" (fruit) differ only by capitalization. Uncased models lowercase everything before processing, destroying this signal. Named entities — especially proper nouns — rely heavily on case for disambiguation.

3. Expect significantly worse performance. Pretrained spaCy models are trained on news and general web text. Medical records use domain-specific terminology, abbreviations, and entity types (drug names, dosages, symptoms, conditions) that the general model has never seen in the right context. You'd need to fine-tune on medical NER datasets (e.g., i2b2, n2c2) or use a medical-specific model like BioBERT.

4. Text classification uses standard accuracy/precision/recall/F1 at the **sample level** (each document gets one prediction). NER uses **entity-level** F1 — a prediction is correct only if both the span boundaries AND the entity type match exactly. Getting the type right but missing one word of the entity span counts as wrong in strict evaluation.

5. Choose LLM extraction when: (1) you need custom entity types not in standard labels and have no training data, (2) the "entities" require complex reasoning (e.g., "extract all obligations that must be completed within 30 days"), (3) entities are context-dependent in complex ways, or (4) you're prototyping and can't afford data collection/training time. Use fine-tuned BERT when you need high throughput (1000+ predictions/second), low latency, or the lowest cost per prediction.

</details>

---

## 12. Go Deeper

- **[spaCy Course (free, interactive)](https://course.spacy.io/)** — the best hands-on NER tutorial. Build a complete NER pipeline from scratch in your browser in ~3 hours. Covers training custom entity types.
- **["BERT: Pre-training of Deep Bidirectional Transformers" (Devlin 2018)](https://arxiv.org/abs/1810.04805)** — Section 4.3 covers NER specifically. See how they adapted BERT for token classification and got SOTA on CoNLL-2003.
- **[CoNLL-2003 NER Dataset Paper](https://aclanthology.org/W03-0419/)** — the standard benchmark dataset. Reading the annotation guidelines gives you intuition for where NER is hard (abbreviations, ambiguous entities).
- **[Hugging Face Token Classification Guide](https://huggingface.co/docs/transformers/tasks/token_classification)** — step-by-step fine-tuning of BERT for NER. Covers the tricky subword alignment issue with worked code.
- **[Prodigy (annotation tool)](https://prodi.gy/)** — if you need to create training data for custom NER, this is the industry-standard annotation tool. Annotate NER examples efficiently with active learning.
