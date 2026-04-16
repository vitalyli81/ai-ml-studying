# Prompt Engineering

## What Is It?

Prompt engineering is the art and science of **writing instructions that get LLMs to do exactly what you want**. It's the most important skill for an AI engineer — the difference between a useless response and a perfect one is often just how you asked.

```
Bad prompt:   "Write code"
Good prompt:  "Write a TypeScript function that validates an email address.
               Return true/false. Include edge cases for subdomains and plus addressing.
               Add JSDoc comments."

Same model, wildly different results. The prompt is the product.
```

## Frontend Analogy

```javascript
// Prompt engineering is like writing a really detailed Jira ticket

// Bad Jira ticket:
//   "Fix the button"
//   → Developer: which button? fix what? on which page?

// Good Jira ticket:
//   "The submit button on /checkout is disabled after first click even when
//    form validation passes. Expected: re-enabled after failed submission.
//    Steps to reproduce: 1. Fill form  2. Submit with bad card  3. Fix card
//    4. Button stays disabled. See screenshot."
//   → Developer: got it, fixing now.

// Same with LLMs — clarity in, quality out.
```

## The Anatomy of a Prompt

Every prompt to an LLM has these building blocks:

```
┌─────────────────────────────────────────────┐
│  SYSTEM PROMPT (role, rules, constraints)   │
├─────────────────────────────────────────────┤
│  CONTEXT (background info, documents)       │
├─────────────────────────────────────────────┤
│  TASK (what you want done)                  │
├─────────────────────────────────────────────┤
│  FORMAT (how you want the output)           │
├─────────────────────────────────────────────┤
│  EXAMPLES (show, don't just tell)           │
└─────────────────────────────────────────────┘
```

### Example with All Parts

```
SYSTEM: You are a senior TypeScript developer who writes clean, testable code.

CONTEXT: We're building a Next.js e-commerce app. The cart uses Zustand for state.

TASK: Write a function that calculates the total price including:
- Quantity discounts (10%+ off for 5+ items)
- Tax by state
- Free shipping over $50

FORMAT: Return a TypeScript function with proper types. Include unit tests.

EXAMPLE:
  Input: [{ item: "Widget", price: 10, qty: 6 }], state: "CA"
  Output: { subtotal: 60, discount: 6, tax: 4.86, shipping: 0, total: 58.86 }
```

## Core Techniques

### 1. Zero-Shot Prompting

Just ask directly. No examples. Works for simple, well-defined tasks.

```
Prompt: "Classify this review as positive, negative, or neutral:
         'The food was okay but the service was terrible.' "

LLM:    "Negative"
```

**When to use:** Simple tasks where the LLM clearly understands what you want.

### 2. Few-Shot Prompting

Provide examples of the input→output pattern you want. The LLM learns the pattern from your examples.

```
Prompt:
  "Classify these reviews:

   Review: 'Amazing product, works perfectly!' → Positive
   Review: 'It's fine, nothing special.' → Neutral
   Review: 'Broke after 2 days, waste of money.' → Negative

   Review: 'Good value but shipping was slow.' → "

LLM:    "Neutral" (or "Mixed" — it learned the pattern)
```

**When to use:** When the task has a specific format or the LLM gets it wrong with zero-shot.

```javascript
// Frontend analogy: Few-shot is like writing test cases

// Instead of explaining the function requirements in words,
// you show the LLM:
//   expect(classify("Amazing!")).toBe("Positive")
//   expect(classify("Meh")).toBe("Neutral")
//   expect(classify("Terrible")).toBe("Negative")

// The LLM infers the function from the test cases
```

### 3. Chain-of-Thought (CoT) Prompting

Ask the LLM to **think step by step** before answering. This dramatically improves reasoning on complex problems.

```
WITHOUT CoT:
  Prompt: "If a shirt costs $25 and is 20% off, and tax is 8%, what do I pay?"
  LLM:    "$21.60"  ← might be right, might be wrong, who knows

WITH CoT:
  Prompt: "If a shirt costs $25 and is 20% off, and tax is 8%, what do I pay?
           Think step by step."
  LLM:    "Step 1: Original price = $25
           Step 2: Discount = 25 × 0.20 = $5
           Step 3: After discount = 25 - 5 = $20
           Step 4: Tax = 20 × 0.08 = $1.60
           Step 5: Total = 20 + 1.60 = $21.60"
```

**Why it works:** Forcing the model to show its work prevents it from skipping steps and making errors. Each step becomes context for the next step.

```javascript
// Frontend analogy: It's like debugging with console.log

// Without CoT: the function returns wrong answer, you have no idea why
// With CoT:    you console.log each intermediate value, and can spot where
//              the logic goes wrong

// console.log("price:", price);          // Step 1
// console.log("discount:", discount);     // Step 2
// console.log("afterDiscount:", after);   // Step 3
// console.log("tax:", tax);               // Step 4
// console.log("total:", total);           // Step 5
```

### 4. System Prompts (Role Prompting)

Set the LLM's persona, rules, and constraints. This is the most powerful lever for consistent behavior.

```
System: "You are a code reviewer for a TypeScript codebase.
         Rules:
         - Focus on bugs, security issues, and performance
         - Ignore style/formatting (we have ESLint for that)
         - Rate severity as: 🔴 critical, 🟡 warning, 🟢 suggestion
         - Be concise — one line per issue
         - If the code looks good, just say 'LGTM'"

User: [pastes code]

LLM: "🔴 Line 15: SQL injection — user input passed directly to query
      🟡 Line 23: N+1 query — this runs inside a loop, use batch fetch
      🟢 Line 31: Consider using Map instead of Object for dynamic keys"
```

### 5. Output Format Control

Tell the LLM exactly what format you need. This is critical for building apps that parse LLM responses.

```
Prompt: "Extract the following from this job posting and return as JSON:
         - title (string)
         - company (string)
         - salary_min (number or null)
         - salary_max (number or null)
         - remote (boolean)
         - skills (string array)

         Job posting: [paste here]"

LLM returns:
{
  "title": "Senior Frontend Engineer",
  "company": "Acme Inc",
  "salary_min": 150000,
  "salary_max": 200000,
  "remote": true,
  "skills": ["React", "TypeScript", "GraphQL"]
}
```

```javascript
// This is huge for AI engineering — you can now do:
const result = JSON.parse(llmResponse);
// and use it directly in your app!

// Pro tip: Anthropic and OpenAI both support "structured outputs"
// which GUARANTEE valid JSON matching a schema — no parsing errors ever
```

## Advanced Techniques

### Prompt Chaining

Break complex tasks into a pipeline of simpler prompts, where each step feeds into the next.

```
Complex task: "Analyze this codebase and suggest refactoring"

Chain:
  Step 1: "List all files in this codebase and their purpose" → file_list
  Step 2: "Given these files, identify code duplication" → duplicates
  Step 3: "Given these duplicates, suggest specific refactoring" → plan
  Step 4: "Implement refactoring step 1 from this plan" → code

Each step is simple and focused. The chain produces better results than
one giant prompt asking for everything at once.
```

```javascript
// Frontend analogy: It's like middleware in Express/Next.js

// Instead of one massive handler:
app.post('/process', (req, res) => { /* 500 lines of everything */ });

// You chain focused middleware:
app.post('/process', validate, enrich, transform, respond);

// Same idea — each LLM call does one thing well
```

### Self-Consistency (Multiple Samples)

Ask the same question multiple times and take the majority answer. Great for reasoning tasks.

```javascript
// Ask the LLM 5 times with temperature > 0
const answers = await Promise.all([
  askLLM(prompt), // "42"
  askLLM(prompt), // "42"
  askLLM(prompt), // "38"  ← outlier
  askLLM(prompt), // "42"
  askLLM(prompt), // "42"
]);

// Majority vote: "42" (4/5 times)
// More confident than a single sample
```

### ReAct (Reason + Act)

The model alternates between **thinking** and **acting** (calling tools). This is the foundation of AI agents.

```
User: "What's the weather in the city where the Eiffel Tower is?"

LLM Thought: I need to know which city the Eiffel Tower is in. I know it's Paris.
             Now I need the weather in Paris.
LLM Action:  call weather_api("Paris, France")
Observation: {"temp": 18, "condition": "partly cloudy"}
LLM Thought: I have the weather data. I can now answer.
LLM Answer:  "The weather in Paris (where the Eiffel Tower is) is 18°C
              and partly cloudy."
```

We'll cover this in depth in [Agents & Tool Use](agents-tool-use.md).

## Prompt Engineering Patterns for AI Engineers

### The System Prompt Template

This is a production-ready template for building AI features:

```
You are [ROLE] that helps users [GOAL].

## Rules
- [Constraint 1]
- [Constraint 2]
- [Safety rule]

## Input Format
You will receive [description of input].

## Output Format
Respond with [exact format specification].
[Include an example]

## Examples
Input: [example input]
Output: [example output]

Input: [edge case input]
Output: [edge case output]

## Edge Cases
- If [situation], then [behavior]
- If unsure, [fallback behavior]
```

### Common Mistakes to Avoid

```
❌ Being vague:        "Make it better"
✅ Being specific:     "Reduce the function to under 20 lines while keeping
                        all edge cases. Use early returns instead of nesting."

❌ Asking for too much: "Build me an entire e-commerce platform"
✅ Breaking it down:    "Write the cart total calculation function"

❌ No format spec:     "Summarize this article"
✅ With format:        "Summarize this article in 3 bullet points,
                        each under 15 words"

❌ Ignoring edge cases: "Parse this date"
✅ Handling edges:      "Parse this date. If ambiguous (01/02/03), assume
                         MM/DD/YY. If invalid, return null with an error message."
```

### Prompt Debugging Checklist

When the LLM gives bad output:

```
1. Is the task clear?          → Can a human understand what you want?
2. Is there enough context?    → Does the LLM have the info it needs?
3. Are examples provided?      → Show don't just tell
4. Is the format specified?    → JSON? Markdown? Bullet points?
5. Are edge cases handled?     → What if input is empty? Invalid?
6. Is the prompt too long?     → Shorter, focused prompts often work better
7. Is temperature appropriate? → Code/facts=0, creative=0.7+
```

## Real-World Example: Building a Code Review Bot

```
System Prompt:
"You are an automated code reviewer for a TypeScript/React codebase.

Rules:
- Only flag real bugs and security issues, not style preferences
- Max 5 comments per review
- Each comment must reference a specific line number
- Rate severity: critical (blocks merge), warning (should fix), info (nice to have)
- If the code is clean, respond with exactly: {"comments": [], "summary": "LGTM"}

Output Format (JSON):
{
  "comments": [
    {
      "line": number,
      "severity": "critical" | "warning" | "info",
      "message": "string",
      "suggestion": "string (optional fix)"
    }
  ],
  "summary": "One-sentence overall assessment"
}

Example:
Input: function divide(a, b) { return a / b; }
Output: {
  "comments": [{
    "line": 1,
    "severity": "critical",
    "message": "Division by zero not handled",
    "suggestion": "if (b === 0) throw new Error('Division by zero')"
  }],
  "summary": "One critical bug: missing zero-division check"
}"
```

## Key Takeaways

| Technique | When to Use | Complexity |
|-----------|-------------|------------|
| Zero-shot | Simple, clear tasks | ⭐ |
| Few-shot | Need specific format/pattern | ⭐⭐ |
| Chain-of-thought | Reasoning, math, logic | ⭐⭐ |
| System prompts | Any production use case | ⭐⭐ |
| Prompt chaining | Complex multi-step tasks | ⭐⭐⭐ |
| Self-consistency | High-stakes decisions | ⭐⭐⭐ |
| ReAct | Tasks requiring external data | ⭐⭐⭐⭐ |

## What's Next?

Great prompts are powerful, but LLMs are limited to what they were trained on. What if you need them to answer questions about **your** data? That's [RAG (Retrieval-Augmented Generation)](rag.md).
