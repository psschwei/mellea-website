---
title: "Making Small Models Rock with Mellea"
date: "2026-06-05"
author: "Paul Schweigert, Nathan Fulton"
excerpt: "Small open-weight models can match frontier-model output on real tasks, if the harness around them is doing its share of the work. Here's how Mellea makes that trade possible."
tags: ["granite", "rag", "adapters", "small-models", "docling", "local-llm"]
---

<img src="/images/small-models-rock/main.png" alt="Making Small Models Rock with Mellea" style="background-color: white;" />

Small open-weight models keep getting better. A 3B Granite model on your
laptop today does things a 70B model couldn't do a year ago. There is still
a gap between "small model can technically do this" and "small model can
reliably do this on real data in production," and most teams cross that gap
by giving up and paying for a frontier reasoning model instead.

They shouldn't have to. The reason a small model fails on most production
tasks isn't raw capability; it's that the small model can't hold a
six-subtask problem in one shot. A frontier model given a thousand-token
system prompt with six subtasks inlined will muddle through. A 3B model
given the same prompt won't.

The fix is a better harness around the model you already have. By
"harness" we mean the software scaffolding around the model call:
decomposition, validation, retries, tool dispatch. The part that isn't
the forward pass. That's what Mellea is for.

> **What this post does**: walks through a construction cost-estimation
> pipeline that one-shot prompting needs a frontier reasoning model for,
> rebuilt on a 3B Granite model running locally. Same accuracy, no API
> keys, ~$0/run. If you're paying frontier-model prices for structured
> extraction or matching, the same pattern applies.

## The Bet

Mellea is built around one observation: the software harness around a model
matters as much as the model itself. Frontier labs already know this. The
team training OpenAI's models knows exactly what the ChatGPT and Codex
harnesses look like, and that knowledge shapes the training data. The
open-weight ecosystem has mostly not had that feedback loop. You upload
weights to Hugging Face and people do whatever with them.

Mellea is the harness side of that co-design. It takes tasks that currently
demand a frontier model, breaks them into pieces a small model can do well,
and holds those pieces together with ordinary code rather than an
ever-growing English prompt.

Three things fall out of that approach:

- **Cost is predictable.** Local inference on a small model has a fixed,
  knowable cost per run, so you can back-test millions of runs without a
  finance conversation.
- **Data stays local.** Documents never leave the machine. For regulated
  domains like healthcare, finance, legal, and procurement, that's the
  difference between "can use this" and "can't."
- **The inference backend is yours to choose.** Mellea talks to Ollama,
  vLLM, Hugging Face, and any OpenAI-compatible endpoint, so swapping
  backends is a one-line change. You aren't married to one provider's
  pricing or uptime.

## The Three Patterns

Mellea turns a task that needs a frontier model into one a small model can
handle through three moves:

![Decompose, externalize control flow, and modularize capabilities](/images/small-models-rock/three-steps.png)

1. **Decompose the task.** A single prompt bundles parsing, fuzzy matching,
   arithmetic, and generation. Split it into steps narrow enough that a
   small model can nail each one.
2. **Externalize control flow.** Keep the `for` loops, `if` statements, and
   retry logic in Python, not in the prompt. Models don't need to be
   instructed to iterate; code iterates.
3. **Modularize model capabilities.** Reuse validated components like
   structured output, typed generative functions, and RAG adapters
   instead of rebuilding them every time.

None of these is new. They are the ordinary software-engineering moves
you'd make for any non-trivial system. What's new is applying them to the
part of your program that happens to be written in English.

The argument in [*On the Foolishness of "Natural Language
Programming"*](https://www.cs.utexas.edu/~EWD/transcriptions/EWD06xx/EWD667.html)
from 1978 applies directly:

> English is designed for communication between humans, not for
> specifying processes executed by machines. The fact that a modern
> model can *interpret* English well enough to try doesn't mean English
> is a good language to write your program in.

Drawing boundaries, stating contracts, decomposing work: that's where
the engineering has always lived. The syntax of the target language
was the trivial part.

## An Example Worth Walking Through

Pick a task that's just past what single-shot prompting can do reliably,
then build it up as a small-model pipeline.

A construction management firm wants to estimate material costs for a
project. The inputs are a construction plan (PDF, with a bill of materials
spread across several tables) and a handful of supplier catalogs (PDF,
DOCX, XLSX). The output is a priced HTML report with a pie chart breaking
spend down by category.

Ask a frontier model to do this in one shot and you can make it work. A
top-tier reasoning model with extended thinking gets most of the line
items right, cites the catalog it pulled the price from, and does it all
for about a dollar per run. Here's what one-shot prompting looks like
across the size spectrum:

| Model | Result |
| --- | --- |
| Small open-weight (<3B) | Doesn't understand the task. Returns a generic "cost breakdown" with no prices. |
| Open-weight reasoning (~20B) | Finds categories and subtotals. No pie chart. Numbers often wrong. |
| Gemini Fast | Mostly reasonable. No chart. Some prices off. |
| Frontier reasoning (GPT-5) | Gets most items right. Cites sources. No chart on first shot. ~$1/run. |

A dollar per run sounds cheap until you write down what the firm actually
wants to do with it. Fifteen hundred projects a year, twenty years of
history, a backlog of hypothetical backtests: *what if we'd switched lumber
suppliers in 2012? What happens to margins if this supply chain constraint
materializes?* One backtest is a Monte Carlo simulation over tens of
thousands of projects. At a dollar a run, that's the budget for a small
team.

Same task, on a 3B Granite model running on a laptop. About a
hundred lines of code. No API keys, no egress, no per-token billing. Here's
how.

## Setup

Install Mellea with the extras this tutorial uses, grab the sample
construction plans and supplier catalogs, then start a session backed by a
local Granite model. The full runnable walkthrough lives in the [tutorial
notebook](https://github.com/generative-computing/mellea-tutorials/blob/main/notebooks/atai_2026/tutorial.ipynb).

```bash
uv pip install 'mellea[docling,hf]'
wget https://nfulton.org/atai26.tar.gz
tar xvfz atai26.tar.gz
```

```python
import mellea
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B

m = mellea.start_session(backend_name="ollama", model_id=IBM_GRANITE_4_MICRO_3B)
```

## Step 1: Parse the Construction Plan into a Bill of Materials

The construction plan is a PDF with tables scattered through it. Some are
bills of materials; others are schedules, notes, or summaries. The first
step is to get a clean, typed `BOM` object.

`RichDocument` wraps [docling](https://github.com/DS4SD/docling) and
exposes tables as markdown, which small models handle much better than raw
HTML or PDF binary:

```python
from mellea.stdlib.components.docs.richdocument import RichDocument

construction_plans = RichDocument.from_document_file(
    "construction_docs/construction_plans.pdf"
)
```

To filter down to material-list tables, Mellea lets you declare a typed
generative function: a Python signature with no body, plus a docstring.

```python
from typing import Literal
import mellea

@mellea.generative
def is_material_list(table_markdown: str) -> Literal["yes", "no"]:
    """Determines if the table contains a list of construction items."""
```

The `@mellea.generative` decorator turns the signature into a constrained
model call. The return type `Literal["yes", "no"]` forces a binary answer at
decode time. The model can't hedge with a paragraph, can't return "maybe,"
and can't return "probably yes." That constraint is enforced by the
sampler, not by a post-hoc prompt rule.

For tables that pass the filter, reformat them into a validated `BOM`:

```python
import pydantic
from mellea.stdlib.requirements import req, simple_validate

class BOMEntry(pydantic.BaseModel):
    item: str
    quantity: int | str
    notes: str
    category: Literal["lumber", "windows", "doors", "other"]

class BOM(pydantic.BaseModel):
    items: list[BOMEntry]

def _bom_is_valid(output: str) -> bool:
    """Quantity is either an integer or the string 'allowance'."""
    bom = BOM.model_validate_json(output)
    return all(
        e.quantity.lower() == "allowance" or str(e.quantity).isdigit()
        for e in bom.items
    )
```

"Allowance" is a construction term of art: *buy some of these, here's twenty
bucks.* The Pydantic schema alone can't express that rule, so it goes into a
requirement with an explicit validator:

```python
m.instruct(
    "Reformat this table to have four columns: item, quantity, category, and notes.",
    grounding_context={"table": table.to_markdown()},
    requirements=[
        req(
            "Quantity should only contain an integer or Allowance",
            validation_fn=simple_validate(_bom_is_valid),
        ),
    ],
    format=BOM,
)
```

Two layers of enforcement: `format=BOM` constrains the decoder to emit valid
JSON matching the schema; the `requirements` list adds business-logic checks
on top. If validation fails, Mellea re-samples. The loop is Python; the model
only has to do the reformatting.

### Parallelize across tables

Construction documents often carry many tables, and each table is
independent of the others. `ainstruct` returns a coroutine that resolves
to a thunk (a deferred computation), so the extraction can fan out:

```python
import asyncio
from mellea.core import ModelOutputThunk

async def extract_bom(doc: RichDocument):
    bom_routines = []
    for table in doc.get_tables():
        if is_material_list(m, table_markdown=table.to_markdown()) == "yes":
            bom_routines.append(m.ainstruct(
                "Reformat this table to have four columns: item, quantity, category, and notes.",
                grounding_context={"table": table.to_markdown()},
                requirements=[
                    req(
                        "Quantity should only contain an integer or Allowance",
                        validation_fn=simple_validate(_bom_is_valid),
                    ),
                ],
                format=BOM,
            ))

    bom_thunks: list[ModelOutputThunk] = await asyncio.gather(*bom_routines)
    boms = [BOM.model_validate_json(await t.avalue()) for t in bom_thunks]

    all_items = []
    for bom in boms:
        all_items.extend(bom.items)
    return BOM(items=all_items)
```

Wall-clock scales with the slowest table, not the sum. The `for`-loop
shape stays in Python; each iteration issues work and collects it later.

## Step 2: Load the Product Catalogs

Catalogs come in whatever format the supplier happened to send. `RichDocument`
handles PDF, DOCX, and XLSX the same way:

```python
from mellea.stdlib.components.docs import Document
from mellea.stdlib.components.docs.richdocument import RichDocument

rd_doors = RichDocument.from_document_file("product_catalogs/door_product_catalog.pdf")
doors_doc = Document(text=rd_doors.to_markdown())

rd_windows = RichDocument.from_document_file("product_catalogs/north_ridge_windows.docx")
windows_doc = Document(text=rd_windows.to_markdown())
```

Each becomes a plain `Document`, the input format Mellea's RAG adapters
expect. Lumber is skipped here to keep the walkthrough short.
Any item whose category isn't keyed into the catalog lookup below will
land in the report as "unknown."

## Step 3: Match BOM Entries to Prices with RAG Adapters

Matching "36x80 Aurora half-moon entry door" from a bill of materials to
the right line in a product catalog is fuzzy by nature. Rather than ask a
general-purpose model to "find the right price and tell me if you're sure,"
Mellea exposes purpose-built adapters from the [IBM Granite RAG adapters
library](https://huggingface.co/ibm-granite) for exactly this kind of check:
`check_context_relevance`, `check_answerability`, and `find_citations`.
(In the Python import path they're still under `intrinsic`; the
externally-facing term is *adapter*.)

The adapters run on the HuggingFace backend, so start a second session.
The `start_session` convenience constructor covers the common Ollama path;
for HuggingFace plus a local backend object, drop one level and construct
`MelleaSession` directly:

```python
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B

m_hf = mellea.MelleaSession(
    backend=LocalHFBackend(model_id=IBM_GRANITE_4_MICRO_3B)
)
```

**Context relevance** asks whether a document is in the right domain for a
question:

```python
from mellea.stdlib.components.intrinsic.rag import (
    check_context_relevance,
    check_answerability,
)
from mellea.stdlib.context import ChatContext

score = check_context_relevance(
    "What is the price of the 36x80 Aurora half-moon entry door?",
    document=doors_doc,
    context=ChatContext(),
    backend=m_hf.backend,
)
```

**Answerability** asks the harder question: given this document, can the
specific question actually be answered?

```python
verdict = check_answerability(
    question="What is the price of the 36x80 Aurora half-moon entry door?",
    documents=[doors_doc],
    context=ChatContext(),
    backend=m_hf.backend,
)
# verdict is the categorical label 'answerable' or 'unanswerable'
```

What makes these adapters worth reaching for, as opposed to asking a
general model "is this document relevant, 1-10," is that the outputs are
**calibrated**. Pass the doors catalog and a doors question, you get a
high-confidence "answerable." Pass the lumber catalog with the same doors
question, and context relevance comes back `"partially relevant"` (a
pricing document about construction, but not the right one) while
answerability correctly collapses to `"unanswerable"`. Frontier model
logits don't give you this. Nothing in a general-purpose model's training
signal makes the raw next-token probabilities mean "calibrated confidence
that this document answers this question." Adapters are trained to make
them mean that.

### Why adapters are cheap to compose

A realistic pipeline wants to run multiple checks on the same
(question, document) pair: is it relevant, is it answerable, where are the
citations? Naively, three adapters means three full model runs, each
re-prefilling the same long document prompt.

Granite adapters ship as **[ALoRA](https://arxiv.org/abs/2504.12397)**
adapters, not plain LoRAs. The difference is narrow but matters here: an
ALoRA reuses the base model's prefill. The expensive pass over the document happens once against the
base weights, and each adapter only kicks in during token generation. Three
sequential checks on the same document share one prefill. You get
modularity without paying 3× the compute.

### The pricing loop

The one line that matters is `if verdict == "answerable":`. Everything
around it is ceremony to get that gate into place. The loop walks every
BOM item, picks the right catalog by category, and asks the answerability
adapter whether a price is extractable before it ever asks the model to
produce one:

```python
class BomEntryWithPrice(pydantic.BaseModel):
    item: str
    quantity: int | str
    notes: str
    category: Literal["lumber", "windows", "doors", "other"]
    unit_price: float | None
    total_price: float | None

class UnitPriceResponseFmt(pydantic.BaseModel):
    unit_price: float

class TotalPriceResponseFmt(pydantic.BaseModel):
    total_price: float

def get_prices(bom: BOM) -> list[BomEntryWithPrice]:
    prices: list[BomEntryWithPrice] = []
    for entry in bom.items:
        # .get() returns None for categories we didn't key in (e.g. lumber),
        # which falls through to the "unknown price" branch below.
        catalog = {"windows": windows_doc, "doors": doors_doc}.get(entry.category)

        if catalog:
            # The gate: don't ask for a price unless the adapter is confident
            # one can be extracted from this document.
            verdict = check_answerability(
                f"What is the price of {entry.item}?",
                documents=[catalog],
                context=ChatContext(),
                backend=m_hf.backend,
            )
            if verdict == "answerable":
                unit = m.instruct(
                    f"Find the `unit_price` of {entry.item} in the catalog.",
                    grounding_context={"catalog": catalog.text},
                    format=UnitPriceResponseFmt,
                )
                unit_price = UnitPriceResponseFmt.model_validate_json(unit.value).unit_price

                total = m.instruct(
                    f"Find the `total_price` given the `unit_price` and `quantity` for {entry.item}",
                    grounding_context={
                        "unit_price": str(unit_price),
                        "quantity": str(entry.quantity),
                    },
                    format=TotalPriceResponseFmt,
                )
                total_price = TotalPriceResponseFmt.model_validate_json(total.value).total_price

                prices.append(BomEntryWithPrice(
                    item=entry.item, quantity=entry.quantity, notes=entry.notes,
                    category=entry.category, unit_price=unit_price, total_price=total_price,
                ))
                continue  # happy path done — skip the fallback append

        # Either no catalog for this category or the adapter said "unanswerable":
        # record the item with unit_price=None so the report surfaces it as unknown.
        prices.append(BomEntryWithPrice(
            item=entry.item, quantity=entry.quantity, notes=entry.notes,
            category=entry.category, unit_price=None, total_price=None,
        ))
    return prices
```

`verdict == "answerable"` is a gate: items the adapter can't confidently
answer don't get a hallucinated price, they get `total_price=None` and
flow through to the report as "unknown." Failure modes are explicit. And
the model is only ever asked to extract a number from a document where a
calibrated adapter has already confirmed the number is there. The hard
problem (*is this the right document at all?*) was solved by a
specialized adapter rather than by a general prompt.

For extra belt-and-suspenders, `find_citations` will highlight the exact
span of the source document that produced each answer, so spot-checks
become a substring lookup rather than a re-read. It works on any response
against any document — here's the shape against a single item:

```python
from mellea.stdlib.components.intrinsic.rag import find_citations

total = m.instruct(
    "Find the `total_price` of the 36x80 Aurora half-moon entry door.",
    grounding_context={"catalog": doors_doc.text},
    format=TotalPriceResponseFmt,
)

citations = find_citations(
    response=total.value,
    documents=[doors_doc],
    context=ChatContext(),
    backend=m_hf.backend,
)
```

## Step 4: Generate the Report

The final step is two instructions: make the chart, then write the report
around it.

Pricing runs on a 3B Granite model, but chart-drawing code is a different
beast. Tool-calling and matplotlib benefit from a model with more muscle,
so for this step switch sessions to GPT-OSS 20B running locally on Ollama.
The same `m` name now points at the larger model for the remainder of the
pipeline:

```python
import json
from mellea.backends.model_ids import OPENAI_GPT_OSS_20B
from mellea.backends.model_options import ModelOption
from mellea.stdlib.tools import local_code_interpreter
from mellea.backends.tools import MelleaTool

m = mellea.start_session(backend_name="ollama", model_id=OPENAI_GPT_OSS_20B)

report_grounding_context = {
    x.item: json.dumps({
        "total_price": x.total_price if x.total_price is not None else "unknown",
        "category": x.category,
    })
    for x in prices
}

pie_chart_result = m.instruct(
    "Use the code interpreter tool to create a pie chart of known cost "
    "breakdowns by category. Put the pie chart in /tmp/chart.png",
    grounding_context=report_grounding_context,
    tool_calls=True,
    model_options={ModelOption.TOOLS: [MelleaTool.from_callable(local_code_interpreter)]},
)
pie_chart_result.tool_calls["local_code_interpreter"].call_func()

report = m.instruct(
    "Write an HTML report with a top-line cost breakdown by category and a "
    "line-item material list with prices. At the top include the /tmp/chart.png image.",
    grounding_context=report_grounding_context,
)
with open("/tmp/report.html", "w") as f:
    f.write(report.value)
```

Swapping sessions mid-pipeline is one line. Each step runs against the
model that suits it: tiny and cheap where the task is narrow, bigger only
where the task demands it.

The chart is drawn by actual Python code executing on actual data, not by a
diffusion model hallucinating a plot. The grounding context is a plain
dict, not a vector store. When the context is small and structured and
known at call time, passing it directly is simpler and more predictable
than dragging in retrieval.

## Small Models, Frontier-Level Output

Step back from the construction example. The general shape of the trade:

<img src="/images/small-models-rock/harnessed.png" alt="A small model, harnessed" style="max-width: 60%;" />

A task that one-shot required a frontier model and a dollar per run now
runs on a 3B open-weight model on a laptop. Pricing extraction on Granite 4
Micro 3B via HuggingFace, chart and report generation on GPT-OSS 20B via
Ollama, no API keys, no egress, no per-token billing. Items the pipeline
can't confidently price show up as `unknown` in the output rather than as
hallucinated numbers, so the report is auditable instead of something a
human has to re-check line by line.

The construction case isn't a one-off. The same three-pattern approach
generalizes. In internal evaluations on agent benchmarks the Mellea team
has run (a DB2 database agent and a compliance agent), rewriting large
prompt-based systems as Mellea programs produces meaningful task-completion
gains on large models and lets much smaller open-weight models approach
the accuracy of a baseline several tiers up. Those points of task
completion are the ones you usually can't buy without going up a model
tier. The harness buys them instead.

The practical upshot: "small model or frontier model" is mostly a question
about whether the harness around the model is doing its share, not about
raw capability. If you're paying frontier-model prices for a task
that decomposes cleanly, there's a good chance you're paying for
engineering you haven't done yet.

## Trade-offs

Latency per run is higher than a single API call. Async execution hides
most of the fan-out, but a pipeline will always be slower per-run than one
prompt. The win is in cost, privacy, and auditability, not in tail latency.

Rate limits disappear. A frontier API can rate-limit you mid-backtest;
local inference can't. When a pipeline fails at 3am, debugging a
decomposed Mellea program means finding *which step* went wrong — a single
frontier-model call is a black box. And the obvious alternative to
"better harness + small model" is "fine-tune a small model." Harnessing
is cheaper to iterate on and composes with fine-tuning later if you need
it.

Decomposition takes engineering effort. It's ordinary software work — a
senior engineer can typically port a prompt pipeline in a day or two, and
the GPU or laptop amortizes fast compared to even a week of frontier-API
backtests. It pays back once the task is recurring, privacy matters, or
cost compounds at scale.

Not everything decomposes. Tasks that are genuinely holistic, like
free-form creative writing or long-form reasoning chains that can't be cut
at clean seams, still favor a big model. Know which kind of task you have
before you invest.

## Try It

If you're running structured extraction, matching, classification, or
report-generation pipelines and paying frontier-model prices for it, the
pieces used above are all part of Mellea's standard library. The full
construction tutorial notebook is in the Mellea tutorials repo.

- [Mellea on GitHub](https://github.com/generative-computing/mellea)
- [Granite RAG adapters on Hugging Face](https://huggingface.co/ibm-granite)
- [Construction tutorial notebook](https://github.com/generative-computing/mellea-tutorials/blob/main/notebooks/atai_2026/tutorial.ipynb)
