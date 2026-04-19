# The Ultimate Python Study Plan

A structured, hands-on path from zero to Python mastery — optimized for AI/ML work. Each phase builds on the previous one; don't skip fundamentals.

**Estimated total time:** 6–12 months at ~1 hour/day (faster if you have prior programming experience).

---

## How to Use This Plan

- **Build, don't just read.** Every phase has a project. Code is learned by writing, not by watching.
- **One concept → one small script.** When you learn a new feature, write a 10-line file that uses it. Keep a `scratch/` folder.
- **Read real code.** After Phase 2, read 10 minutes of open-source Python daily (start with `requests`, `flask`, `httpx`).
- **Test what you write.** From Phase 3 onward, write `pytest` tests alongside every script.
- **Use uv or poetry.** Skip `pip install` into global Python. Use isolated environments from day one.

---

## Phase 0 — Setup (1–2 days)

Get a real development environment before writing any code.

- [ ] Install Python 3.12+ via `pyenv` or `uv`
- [ ] Install `uv` (fast package manager) — [astral.sh/uv](https://docs.astral.sh/uv/)
- [ ] Install VS Code or PyCharm + Python extension + Ruff extension
- [ ] Configure `ruff` for linting/formatting (replaces black, isort, flake8)
- [ ] Learn `uv init`, `uv add`, `uv run`, `uv venv`
- [ ] Set up git; create a `python-study` repo for your work

**Deliverable:** a working `uv`-managed project that runs `print("hello")`.

---

## Phase 1 — Core Language (3–4 weeks)

The absolute fundamentals. Don't move on until these are reflex.

### 1.1 Syntax & Types
- Variables, `int`, `float`, `str`, `bool`, `None`
- f-strings, string methods (`.split`, `.join`, `.strip`, `.replace`, `.format`)
- Operators, truthiness, `is` vs `==`
- `type()`, `isinstance()`

### 1.2 Control Flow
- `if` / `elif` / `else`, ternary expressions
- `for` / `while`, `break`, `continue`, `else` on loops
- `match` / `case` (structural pattern matching, 3.10+)

### 1.3 Data Structures
- `list`, `tuple`, `set`, `dict` — when to use each
- List/dict/set comprehensions
- Slicing, unpacking (`a, *rest = [1,2,3]`)
- `collections`: `Counter`, `defaultdict`, `deque`, `namedtuple`

### 1.4 Functions
- `def`, positional/keyword args, `*args`, `**kwargs`
- Default values (and the mutable-default gotcha)
- Return values, multiple returns via tuples
- Scope: local, enclosing, global, built-in (LEGB)
- Lambdas, `map`, `filter`, `sorted(key=...)`

### 1.5 Errors
- `try` / `except` / `else` / `finally`
- Raising exceptions, custom exception classes
- EAFP vs LBYL

**Project:** CLI todo app — add/list/complete/delete, persist to JSON.

**Resources:** [Python Tutorial](https://docs.python.org/3/tutorial/), *Automate the Boring Stuff*, *Python Crash Course*.

---

## Phase 2 — Idiomatic Python (3–4 weeks)

The difference between "writing Python" and "writing Pythonic code."

### 2.1 Iteration Deep-Dive
- Iterators vs iterables, `iter()`, `next()`
- Generators (`yield`), generator expressions
- `itertools`: `chain`, `groupby`, `product`, `combinations`, `islice`
- `enumerate`, `zip`, `zip_longest`

### 2.2 Files & I/O
- `open()` with context managers, reading modes
- `pathlib.Path` (prefer over `os.path`)
- `json`, `csv`, `tomllib` (3.11+)
- stdin/stdout/stderr, `argparse` for CLI args

### 2.3 Modules & Packages
- `import` mechanics, `__name__ == "__main__"`
- Package layout (`src/` vs flat)
- `pyproject.toml`, entry points
- Virtual environments (why they exist, how `uv` handles them)

### 2.4 Functional Patterns
- First-class functions, higher-order functions
- Closures, `nonlocal`
- Decorators (without `@functools.wraps` first, then with it)
- `functools.partial`, `functools.cache`, `functools.reduce`

**Project:** Log analyzer — stream a large log file with a generator, extract stats, output JSON/CSV report.

**Resources:** *Fluent Python* (Ramalho) — chapters 1–9. [PEP 8](https://peps.python.org/pep-0008/). [Python Patterns](https://github.com/faif/python-patterns).

---

## Phase 3 — Object-Oriented & Type-Aware Python (4 weeks)

Type hints and classes are now standard in any serious codebase.

### 3.1 Classes
- `class`, `__init__`, `self`
- Instance vs class attributes
- Methods, `@classmethod`, `@staticmethod`, `@property`
- Inheritance, `super()`, MRO
- Dunder methods: `__repr__`, `__eq__`, `__hash__`, `__lt__`, `__len__`, `__iter__`, `__enter__`/`__exit__`

### 3.2 Modern Class Patterns
- `@dataclass` (frozen, slots, kw_only)
- `enum.Enum`, `enum.StrEnum` (3.11+)
- Protocols (structural typing) vs ABCs
- `pydantic` v2 (the de facto data modeling library)

### 3.3 Type Hints
- Built-in generics: `list[int]`, `dict[str, int]`
- `Optional`, `Union` / `X | Y` syntax
- `TypedDict`, `NamedTuple`, `Literal`, `Final`
- `TypeVar`, generic functions and classes
- `Protocol`, `@runtime_checkable`
- Run `mypy` or `pyright` on every project

### 3.4 Testing
- `pytest` — fixtures, parametrize, markers
- `pytest-cov` for coverage
- Mocking with `unittest.mock` / `pytest-mock`
- Property-based testing with `hypothesis` (intro)

**Project:** Library management system — classes for Book/Member/Loan, full type hints, 80%+ test coverage.

**Resources:** *Fluent Python* chapters 11–15. [mypy docs](https://mypy.readthedocs.io/). [pytest docs](https://docs.pytest.org/).

---

## Phase 4 — Intermediate / Real-World Python (4–6 weeks)

The ecosystem. What professional Python developers use daily.

### 4.1 Concurrency & Parallelism
- Threading vs multiprocessing vs async — when to use which
- The GIL: what it is, what it isn't (and free-threading in 3.13+)
- `concurrent.futures` (`ThreadPoolExecutor`, `ProcessPoolExecutor`)
- `asyncio` basics: `async`/`await`, event loop, `asyncio.gather`, `asyncio.TaskGroup`
- `httpx` for async HTTP

### 4.2 Data Stack
- `numpy` — ndarrays, broadcasting, vectorization
- `pandas` — Series, DataFrame, groupby, joins, I/O
- `polars` (fast alternative to pandas, worth learning)
- `matplotlib` + `seaborn` for plots
- Jupyter notebooks (and when NOT to use them)

### 4.3 Web & APIs
- HTTP basics with `requests` / `httpx`
- Build an API with `FastAPI` (sync + async endpoints, Pydantic models)
- SQL with `sqlite3`, then `SQLAlchemy` 2.0
- Environment config with `pydantic-settings`

### 4.4 Tooling & Quality
- `ruff` (lint + format)
- `mypy` / `pyright` (type checking)
- `pre-commit` hooks
- `pytest` + coverage in CI
- `tox` or `nox` for multi-env testing

**Project:** FastAPI service with SQLAlchemy, migrations (`alembic`), full test suite, Dockerfile, GitHub Actions CI.

**Resources:** [FastAPI docs](https://fastapi.tiangolo.com/), [SQLAlchemy 2.0 docs](https://docs.sqlalchemy.org/), [Real Python](https://realpython.com/).

---

## Phase 5 — Advanced Python (ongoing)

The stuff that separates senior engineers. Learn as needed, not all at once.

### 5.1 Language Internals
- Descriptors (how `@property` works underneath)
- Metaclasses (and why you almost never need them)
- `__slots__` for memory optimization
- Context managers via `contextlib` (`@contextmanager`, `ExitStack`)
- Reference counting, garbage collection, weak references

### 5.2 Performance
- Profiling: `cProfile`, `py-spy`, `scalene`
- `timeit` for microbenchmarks
- Vectorization with numpy; when to drop to C
- `Cython` / `mypyc` / `numba` — when each makes sense
- PyPy for pure-Python speedups

### 5.3 Async Mastery
- Structured concurrency (TaskGroups, `anyio`)
- Async context managers and async generators
- Backpressure, cancellation, timeouts
- `trio` (alternative async framework — study for ideas)

### 5.4 Packaging & Distribution
- `pyproject.toml` deeply: build backends, dependency groups
- Publishing to PyPI
- C extensions with `pybind11` / `maturin` (Rust)
- Lockfiles: `uv.lock`, `poetry.lock`

**Project:** Open-source a small utility library on PyPI with full docs, CI, and type stubs.

**Resources:** *Fluent Python* chapters 16–24. *High Performance Python* (Gorelick & Ozsvald). [Talks by Raymond Hettinger](https://www.youtube.com/results?search_query=raymond+hettinger).

---

## Phase 6 — Python for AI/ML (ongoing, parallel to Phase 4+)

Since this is an AI/ML studying repo, this track runs alongside the later phases.

### 6.1 Numerical Foundations
- `numpy` fluency (broadcasting, einsum, random)
- `scipy` for optimization, stats, linear algebra

### 6.2 Classical ML
- `scikit-learn`: pipelines, cross-validation, model selection
- Feature engineering with `pandas` + `sklearn`

### 6.3 Deep Learning
- `pytorch` — tensors, autograd, `nn.Module`, training loops
- `torch.utils.data` Datasets and DataLoaders
- GPU basics (`.to(device)`, mixed precision)
- `lightning` or plain PyTorch — pick one and go deep

### 6.4 LLMs & Modern AI
- `transformers` (Hugging Face)
- Tokenizers, embeddings, attention intuition
- `anthropic` / `openai` SDKs; prompt engineering in code
- Vector DBs: `chroma`, `qdrant`, `pgvector`
- RAG pipelines, agent frameworks
- Fine-tuning basics (LoRA, PEFT)

### 6.5 MLOps Lite
- Experiment tracking: `mlflow` or `wandb`
- Model serving: `fastapi` + `onnx` / `torchserve`
- Reproducibility: seeds, `uv.lock`, data versioning (`dvc`)

**Projects:**
1. Train a classifier end-to-end with sklearn; deploy via FastAPI.
2. Fine-tune a small transformer on a custom dataset.
3. Build a RAG chatbot with Anthropic's API + a vector DB.

---

## Habits That Compound

These matter more than any specific tutorial:

1. **Read the docs first.** `help()`, `dir()`, official docs — before Stack Overflow.
2. **Write it, then improve it.** First make it work, then make it clean, then make it fast.
3. **Read tracebacks bottom-up.** The last frame is usually where *your* code failed.
4. **Use `ipython` or `ptpython`** for interactive exploration.
5. **Learn your debugger.** `breakpoint()` (Python 3.7+) beats `print()` past a certain point.
6. **Keep a "gotchas" note.** Every surprising bug, write it down. Mutable defaults. Late binding in closures. `is` on small ints. You'll stop repeating them.
7. **Ship small things.** A 200-line script that works beats a 2000-line project that doesn't.

---

## Reference Library

**Books (in order of usefulness):**
- *Python Crash Course* — Matthes (beginner)
- *Fluent Python, 2nd ed.* — Ramalho (the one to own)
- *Effective Python, 3rd ed.* — Slatkin (90 concrete best practices)
- *High Performance Python* — Gorelick & Ozsvald (when you care about speed)
- *Architecture Patterns with Python* — Percival & Gregory (design)

**Sites:**
- [docs.python.org](https://docs.python.org/3/) — always the first stop
- [Real Python](https://realpython.com/) — polished tutorials
- [PEP Index](https://peps.python.org/) — read PEP 8, 20, 257, 484, 517, 621
- [Python Discord](https://pythondiscord.com/) — active help community

**Talks to watch:**
- Raymond Hettinger — "Transforming Code into Beautiful, Idiomatic Python"
- David Beazley — anything on generators, concurrency, or the GIL
- James Powell — "So you want to be a Python expert?"

---

## Self-Assessment Milestones

You can call a phase "done" when you can do all of these without Googling:

- **Phase 1:** Solve any basic problem on [Exercism Python track](https://exercism.org/tracks/python) or easy LeetCode.
- **Phase 2:** Rewrite an imperative script using generators + comprehensions + decorators.
- **Phase 3:** Design a small class hierarchy with full type hints that passes `mypy --strict`.
- **Phase 4:** Stand up a tested FastAPI service with a database in under a day.
- **Phase 5:** Profile a slow script and speed it up by 10×.
- **Phase 6:** Take a raw dataset to a deployed model (even a simple one) end-to-end.

Good luck. Python rewards curiosity — stay curious.
