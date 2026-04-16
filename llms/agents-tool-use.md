# Agents & Tool Use

## What Is It?

An AI agent is an LLM that can **take actions** in the real world — not just generate text, but actually call functions, search the web, query databases, write code, and execute multi-step plans. Tool use (also called "function calling") is the mechanism that makes this possible.

```
Regular LLM:
  User: "What's the weather in Paris?"
  LLM:  "I don't have real-time weather data." (can only use training knowledge)

LLM with Tools:
  User: "What's the weather in Paris?"
  LLM:  → calls get_weather("Paris")
        → receives { temp: 18, condition: "sunny" }
  LLM:  "It's 18°C and sunny in Paris right now!"
```

## Frontend Analogy

```javascript
// An AI agent is like a React component with side effects

// Regular component (pure LLM):
function StaticInfo() {
  return <div>Paris is the capital of France</div>;  // Static knowledge only
}

// Component with side effects (Agent):
function WeatherWidget() {
  const [weather, setWeather] = useState(null);
  
  useEffect(() => {
    // This is like "tool use" — the component can fetch external data
    fetch('/api/weather?city=Paris')
      .then(res => res.json())
      .then(setWeather);
  }, []);

  return <div>Paris: {weather?.temp}°C, {weather?.condition}</div>;
}

// The LLM decides WHEN to call which function, just like
// your component decides when to trigger useEffect
```

## How Tool Use Works

The flow has 4 steps:

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 1. You   │     │ 2. LLM   │     │ 3. You   │     │ 4. LLM   │
│ send msg │────►│ decides  │────►│ execute  │────►│ generates│
│ + tools  │     │ to call  │     │ the tool │     │ final    │
│          │     │ a tool   │     │ & return │     │ response │
│          │     │          │     │ result   │     │          │
└──────────┘     └──────────┘     └──────────┘     └──────────┘

You define the tools → LLM chooses which to call → You run them → LLM uses results
```

### Step-by-Step

```typescript
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic();

// Step 1: Define tools the LLM can use
const tools = [
  {
    name: 'get_weather',
    description: 'Get current weather for a city. Use this when users ask about weather.',
    input_schema: {
      type: 'object',
      properties: {
        city: { type: 'string', description: 'City name, e.g. "Paris, France"' },
        units: { type: 'string', enum: ['celsius', 'fahrenheit'], default: 'celsius' }
      },
      required: ['city']
    }
  },
  {
    name: 'search_web',
    description: 'Search the web for current information.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' }
      },
      required: ['query']
    }
  }
];

// Step 2: Send message with tools
const response = await anthropic.messages.create({
  model: 'claude-sonnet-4-5-20241022',
  max_tokens: 1024,
  tools,
  messages: [{ role: 'user', content: "What's the weather in Paris?" }]
});

// Step 3: Check if the LLM wants to call a tool
if (response.stop_reason === 'tool_use') {
  const toolCall = response.content.find(c => c.type === 'tool_use');
  // toolCall = { type: 'tool_use', name: 'get_weather', input: { city: 'Paris' } }

  // Step 4: Execute the tool and return the result
  const weatherData = await getWeatherAPI(toolCall.input.city);
  
  // Step 5: Send the result back to the LLM
  const finalResponse = await anthropic.messages.create({
    model: 'claude-sonnet-4-5-20241022',
    max_tokens: 1024,
    tools,
    messages: [
      { role: 'user', content: "What's the weather in Paris?" },
      { role: 'assistant', content: response.content },
      {
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: toolCall.id,
          content: JSON.stringify(weatherData)
        }]
      }
    ]
  });
  
  console.log(finalResponse.content[0].text);
  // "It's currently 18°C and sunny in Paris!"
}
```

## The Agent Loop

Real agents don't just call one tool — they loop: think → act → observe → think → act → ... until the task is done.

```typescript
// The core agent loop pattern
async function agentLoop(userMessage: string) {
  const messages = [{ role: 'user', content: userMessage }];

  while (true) {
    // 1. Ask the LLM what to do
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-5-20241022',
      max_tokens: 4096,
      system: 'You are a helpful assistant with access to tools. Use them when needed.',
      tools,
      messages,
    });

    // 2. Add the assistant's response to history
    messages.push({ role: 'assistant', content: response.content });

    // 3. If the LLM is done (no more tool calls), return the answer
    if (response.stop_reason === 'end_turn') {
      const textContent = response.content.find(c => c.type === 'text');
      return textContent?.text;
    }

    // 4. Otherwise, execute all tool calls
    const toolResults = [];
    for (const block of response.content) {
      if (block.type === 'tool_use') {
        console.log(`Calling tool: ${block.name}(${JSON.stringify(block.input)})`);
        const result = await executeTool(block.name, block.input);
        toolResults.push({
          type: 'tool_result',
          tool_use_id: block.id,
          content: JSON.stringify(result),
        });
      }
    }

    // 5. Send results back and loop
    messages.push({ role: 'user', content: toolResults });
  }
}

// Tool executor — maps tool names to actual functions
async function executeTool(name: string, input: any) {
  switch (name) {
    case 'get_weather': return await fetchWeather(input.city);
    case 'search_web':  return await searchGoogle(input.query);
    case 'run_sql':     return await executeSQL(input.query);
    default: throw new Error(`Unknown tool: ${name}`);
  }
}
```

```javascript
// Frontend analogy: The agent loop is like a state machine

// Think of it as a reducer:
// state = { messages: [], status: 'thinking' }
//
// action: LLM_RESPONSE (with tool_use) → status: 'executing_tools'
// action: TOOL_RESULTS → status: 'thinking'        ← back to LLM
// action: LLM_RESPONSE (end_turn) → status: 'done' ← final answer
//
// The loop keeps going until the LLM says "I'm done"
```

## Multi-Step Agent Example

Here's how an agent handles a complex task with multiple tool calls:

```
User: "Compare the weather in Paris and Tokyo, then find flights between them"

Agent thinking: I need weather for both cities and then flight info.

Step 1: Call get_weather("Paris")      → 18°C, sunny
Step 2: Call get_weather("Tokyo")      → 22°C, cloudy
Step 3: Call search_flights("Paris", "Tokyo") → [list of flights]

Agent: "Here's the comparison:
        Paris: 18°C and sunny
        Tokyo: 22°C and cloudy
        
        Cheapest flights:
        - Air France: $650, 12h direct
        - ANA: $720, 11.5h direct
        - Turkish Airlines: $480, 16h with stopover"
```

The LLM **autonomously decides** the order of operations, what tools to call, and when it has enough information to answer.

## Designing Good Tools

### Tool Description is Everything

The LLM decides which tool to call based on the **description**. Bad descriptions = wrong tool choices.

```typescript
// ❌ Bad tool definition
{
  name: 'db_query',
  description: 'Query the database',  // Too vague — when should LLM use this?
  input_schema: { ... }
}

// ✅ Good tool definition
{
  name: 'search_products',
  description: 'Search the product catalog by name, category, or price range. Use this when the user asks about products, prices, or availability. Returns up to 10 matching products with name, price, and stock status.',
  input_schema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'Search term (product name or keyword)' },
      category: { type: 'string', enum: ['electronics', 'clothing', 'food', 'books'] },
      max_price: { type: 'number', description: 'Maximum price in USD' },
    },
    required: ['query']
  }
}
```

### Tool Design Principles

```
1. CLEAR NAMES       → search_products, not sp or tool_3
2. DETAILED DESC     → When to use it, what it returns, edge cases
3. TYPED PARAMS      → Use enums, required fields, descriptions
4. SINGLE PURPOSE    → One tool per action (don't combine search + buy)
5. SAFE BY DEFAULT   → Read operations first, writes need confirmation
6. USEFUL ERRORS     → Return error messages the LLM can act on
```

## Agent Frameworks

You can build agents from scratch (like above), or use frameworks that handle the loop, memory, and tooling.

### LangChain / LangGraph

```python
# LangChain — popular but heavyweight
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent

llm = ChatAnthropic(model="claude-sonnet-4-5-20241022")
agent = create_tool_calling_agent(llm, tools, prompt)
result = agent.invoke({"input": "What's the weather in Paris?"})
```

### Claude Agent SDK (Anthropic)

```typescript
// For building production agents with Claude
// Handles the agent loop, tool execution, and conversation management
import { Agent } from '@anthropic-ai/agent';

const agent = new Agent({
  model: 'claude-sonnet-4-5-20241022',
  tools: [weatherTool, searchTool, databaseTool],
  system: 'You are a helpful assistant.',
});

const result = await agent.run('Find the cheapest flights to Paris this weekend');
```

### Vercel AI SDK (Tools)

```typescript
// Great for Next.js apps — tools integrate with streaming
import { anthropic } from '@ai-sdk/anthropic';
import { generateText, tool } from 'ai';
import { z } from 'zod';

const result = await generateText({
  model: anthropic('claude-sonnet-4-5-20241022'),
  tools: {
    weather: tool({
      description: 'Get the weather in a city',
      parameters: z.object({
        city: z.string().describe('City name'),
      }),
      execute: async ({ city }) => {
        // Your actual API call
        return await fetchWeather(city);
      },
    }),
  },
  maxSteps: 5,  // Max tool-call iterations
  prompt: "What's the weather in Paris?",
});
```

## Multi-Step Reasoning (ReAct Pattern)

The most powerful agent pattern: **Reason**, **Act**, **Observe** in a loop.

```
User: "What's the total revenue from our top 3 customers this quarter?"

Thought 1: I need to find the top 3 customers by revenue. I'll query the database.
Action 1:  run_sql("SELECT customer, SUM(amount) as revenue 
                     FROM orders 
                     WHERE date >= '2024-10-01' 
                     GROUP BY customer 
                     ORDER BY revenue DESC 
                     LIMIT 3")
Observe 1: [{ customer: "Acme", revenue: 150000 }, 
            { customer: "Beta", revenue: 120000 },
            { customer: "Gamma", revenue: 95000 }]

Thought 2: I have the top 3 customers and their revenue. Let me calculate the total.
Action 2:  (no tool needed — can calculate: 150000 + 120000 + 95000 = 365000)

Answer: "Your top 3 customers this quarter are:
         1. Acme Corp: $150,000
         2. Beta Inc: $120,000  
         3. Gamma Ltd: $95,000
         Total revenue: $365,000"
```

## Agent Safety & Guardrails

Agents can take real actions — so safety is critical.

```typescript
// 1. Permission levels — categorize tools by risk
const readOnlyTools = [searchProducts, getWeather, lookupUser];
const writeTools = [createOrder, updateProfile, sendEmail];
const dangerousTools = [deleteAccount, executeSQL, deployCode];

// 2. Confirmation for write operations
async function executeTool(name: string, input: any) {
  const tool = allTools.find(t => t.name === name);
  
  if (tool.requiresConfirmation) {
    const approved = await askUser(
      `The AI wants to: ${name}(${JSON.stringify(input)}). Allow?`
    );
    if (!approved) return { error: 'User denied this action' };
  }
  
  return await tool.execute(input);
}

// 3. Input validation — never trust LLM-generated inputs blindly
async function executeSql(query: string) {
  // Only allow SELECT queries
  if (!query.trim().toUpperCase().startsWith('SELECT')) {
    return { error: 'Only SELECT queries are allowed' };
  }
  // Use parameterized queries when possible
  return await db.query(query);
}

// 4. Rate limiting — prevent runaway loops
const MAX_ITERATIONS = 10;
let iterations = 0;

while (iterations < MAX_ITERATIONS) {
  iterations++;
  // ... agent loop ...
}
```

```
Safety Hierarchy:
─────────────────
Read operations       → Auto-approve (search, lookup, get)
Write operations      → Require user confirmation (create, update, send)
Destructive operations → Require explicit confirmation + logging (delete, deploy)
Never allow           → Arbitrary code execution without sandbox
```

## Real-World Agent Architectures

### Customer Support Agent

```
Tools:
  - search_knowledge_base(query)     → Find relevant help articles
  - lookup_order(order_id)           → Get order details
  - check_inventory(product_id)      → Check stock status
  - create_ticket(details)           → Escalate to human
  - process_refund(order_id, amount) → Issue refund (requires confirmation)

System prompt:
  "You are a customer support agent. Try to resolve issues using
   available tools. If you can't resolve within 3 tool calls,
   create a ticket and escalate to a human agent."
```

### Code Assistant Agent

```
Tools:
  - read_file(path)           → Read source code
  - search_codebase(query)    → Search files by content
  - run_tests(file)           → Run test suite
  - edit_file(path, changes)  → Modify code
  - run_command(cmd)           → Execute shell commands (sandboxed)

Flow:
  1. User: "Fix the bug in the login form"
  2. Agent: search_codebase("login form") → finds LoginForm.tsx
  3. Agent: read_file("LoginForm.tsx") → sees the code
  4. Agent: identifies bug, edits file
  5. Agent: run_tests("LoginForm.test.tsx") → tests pass
  6. Agent: "Fixed! The issue was..."
```

## Parallel Tool Calls

LLMs can call multiple tools simultaneously when the calls are independent:

```typescript
// The LLM returns multiple tool_use blocks in one response
response.content = [
  { type: 'tool_use', name: 'get_weather', input: { city: 'Paris' } },
  { type: 'tool_use', name: 'get_weather', input: { city: 'Tokyo' } },
  { type: 'tool_use', name: 'search_flights', input: { from: 'Paris', to: 'Tokyo' } }
];

// Execute them in parallel for speed
const results = await Promise.all(
  response.content
    .filter(c => c.type === 'tool_use')
    .map(tc => executeTool(tc.name, tc.input))
);
```

## Key Takeaways

| Concept | What to Remember |
|---------|-----------------|
| Tool use | LLM chooses which function to call based on description |
| Agent loop | Think → Act → Observe → repeat until done |
| Tool design | Clear name, detailed description, typed parameters |
| Safety | Read=auto, Write=confirm, Delete=explicit confirm |
| Parallel calls | Independent tools can run concurrently |
| Max iterations | Always cap the loop to prevent runaway agents |
| ReAct | Reason + Act pattern for complex multi-step tasks |

## What's Next?

Agents use tools to act in the world. But sometimes you need the model itself to behave differently — write in your brand voice, understand your domain terminology, or follow company-specific guidelines. That's [Fine-tuning](fine-tuning-llms.md).
