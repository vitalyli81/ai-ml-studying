Classical Machine Learning

- **Supervised Learning** — linear/logistic regression, decision trees, SVMs, random forests
- **Unsupervised Learning** — clustering (K-means), dimensionality reduction (PCA)
- **Model Evaluation** — cross-validation, bias-variance tradeoff, metrics (precision, recall, F1)
- **Scikit-learn** — hands-on projects with real datasets
- **Feature Engineering** — encoding, scaling, selection techniques

Main file "README.md" is general document for main topic.

Here's a prompt designed for Claude Code to generate teaching-quality docs on any AI topic:

```
You are an expert technical educator creating a learning document on: "Classical Machine Learning"

Create a comprehensive markdown file named `[topic-slug].md` (if not created) that teaches this topic to a frontend engineer transitioning into AI engineering. The reader is smart but new to AI concepts.

## Core Principles (follow strictly)

1. **Feynman technique**: Explain like teaching a curious beginner. If a 12-year-old couldn't grasp the core idea, rewrite it.
2. **Concrete before abstract**: Every concept starts with a real-world analogy or example, THEN the technical definition.
3. **Show, don't just tell**: Use code, diagrams (ASCII/mermaid), and visual comparisons.
4. **Progressive disclosure**: Simple → Intermediate → Advanced. Never dump complexity upfront.
5. **Memory hooks**: Use mnemonics, acronyms, visual metaphors, and "think of it as..." framings.
6. **Active recall prompts**: End sections with quick self-check questions.

## Required Structure

### 1. TL;DR (3-5 lines)
The entire topic in plain English. If someone reads only this, they should "get it."

### 2. The Mental Model (1 paragraph + analogy)
A powerful analogy from everyday life (cooking, building, traffic, etc.) that maps to the topic. Explicitly label the mapping:
- Real world X → Technical concept X
- Real world Y → Technical concept Y

### 3. Why It Exists (Problem → Solution)
- What problem did this solve?
- What came before it and why was it insufficient?
- What changed because of this?

### 4. Core Concepts (the heart of the doc)
Break the topic into 4-8 key concepts. For EACH concept:
- **One-line definition** (plain English)
- **Analogy** (concrete, visual)
- **Technical explanation** (precise but accessible)
- **Minimal code example** or diagram (if applicable)
- **Common misconception** ("People often think X, but actually Y")

### 5. How It Actually Works (Step-by-Step)
Walk through a complete example from input to output. Number every step. Show what happens at each stage. Use a diagram where possible.

### 6. Code in Practice
Provide 2-3 runnable code examples (Python or JS/TS), from simple to realistic. Comment every non-obvious line. Show:
- Minimal "hello world" version
- Practical usage
- A common real-world pattern

### 7. Gotchas & Pitfalls
List 5-7 things that commonly trip people up. Format: "❌ Wrong assumption → ✅ Reality"

### 8. When to Use / When NOT to Use
A clear decision guide. Include 3-4 scenarios for each side.

### 9. Related Concepts (The Map)
How this topic connects to other AI concepts. A brief "if you know X, this is like X but for Y" for 3-5 adjacent topics.

### 10. Cheat Sheet
A dense, scannable summary:
- Key terms + one-line definitions
- Core formula/pattern/API (if applicable)
- "Remember this" — the 3 things that matter most

### 11. Self-Check Questions
5 questions that test understanding (not memorization). Include brief answers in a collapsible `<details>` block.

### 12. Go Deeper
3-5 curated resources: the single best paper, the best blog post, the best video, official docs, and one hands-on tutorial. Explain WHY each is worth the time.

## Formatting Rules

- Use headers, bullet points, tables, and code blocks generously for scannability
- Bold key terms on first use
- Use callout blocks for important notes: `> 💡 **Key Insight:** ...`
- Keep paragraphs to 3 sentences max
- Use tables to compare concepts side-by-side
- Include at least 2 diagrams (mermaid or ASCII)
- Never use jargon without immediately defining it
- Prefer "you" over "one" or "the user"

## Quality Bar

Before finishing, verify:
- [ ] Could a smart beginner read this once and explain the topic to someone else?
- [ ] Is every abstract concept grounded in a concrete example?
- [ ] Are there at least 3 memorable analogies?
- [ ] Does it answer "why should I care?" within the first 200 words?
- [ ] Could I use the cheat sheet alone to refresh my memory in 2 minutes?

Save the file as `[topic-slug].md` in the current directory. (if not created)
```