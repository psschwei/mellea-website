---
title: "Making Small Models Rock with Mellea"
date: "2026-06-05"
author: "Paul Schweigert, Nathan Fulton"
excerpt: "Small open-weight models can match frontier-model output on real tasks, if the harness around them is doing its share of the work. Here's how Mellea makes that trade possible."
tags: ["mellea", "granite", "rag", "intrinsics", "small-models", "docling"]
---

![Making Small Models Rock with Mellea](/images/small-models-rock/main.png)

Small open-weight models keep getting better. A 3B Granite model on your
laptop today does things a 70B model couldn't do a year ago. There is still
a gap between "small model can technically do this" and "small model can
reliably do this on real data in production," and most teams cross that gap
by giving up and paying for GPT-5.4 Pro instead.

They shouldn't have to. The reason a small model fails on most production
tasks isn't raw capability; it's that the small model can't hold a
six-subtask problem in one shot. A frontier model given a thousand-token
system prompt with six subtasks inlined will muddle through. A 3B model
given the same prompt won't.

The fix is a better harness around the model you already have. That's what
Mellea is for.

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

Three things fall out of that approach. The first is predictable cost:
local inference on a small model has a fixed, knowable cost per run, so you
can back-test millions of runs without a finance conversation. The second
is data sovereignty. Your documents never leave the machine, which for
regulated domains like healthcare, finance, legal, and procurement is the
difference between "can use this" and "can't." The third is vendor-agnostic
inference. Mellea talks to Ollama, vLLM, Hugging Face, and any
OpenAI-compatible endpoint, so swapping backends is a one-line change. You
aren't married to one provider's pricing or uptime.

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
   structured output, typed generative functions, and RAG intrinsics
   instead of rebuilding them every time.

None of these is new. They are the ordinary software-engineering moves
you'd make for any non-trivial system. What's new is applying them to the
part of your program that happens to be written in English.

The argument in [*On the Foolishness of "Natural Language
Programming"*](https://www.cs.utexas.edu/~EWD/transcriptions/EWD06xx/EWD667.html)
from 1978 applies directly. English is designed for communication between
humans, not for specifying processes executed by machines. The fact that a
modern model can *interpret* English well enough to try doesn't mean
English is a good language to write your program in. Drawing boundaries,
stating contracts, decomposing work: that's where the actual engineering
has always lived. The syntax of the target language was the trivial part.

## An Example Worth Walking Through

The easiest way to show this working is to pick a task that's just past
what single-shot prompting can do reliably, then build it up as a
small-model pipeline.

A construction management firm wants to estimate material costs for a
project. The inputs are a construction plan (PDF, with a bill of materials
spread across several tables) and a handful of supplier catalogs (PDF,
DOCX, XLSX). The output is a priced HTML report with a pie chart breaking
spend down by category.

Ask a frontier model to do this in one shot and you can make it work.
GPT-5.4 Pro with extended thinking gets most of the line items right,
cites the catalog it pulled the price from, and does it all for about a
dollar per run. Here's what one-shot prompting looks like across the size
spectrum:

| Model | Result |
|---|---|
| Small open-weight (<3B) | Doesn't understand the task. Returns a generic "cost breakdown" with no prices. |
| Open-weight reasoning (~20B) | Finds categories and subtotals. No pie chart. Numbers often wrong. |
| Gemini Fast | Mostly reasonable. No chart. Some prices off. |
| GPT-5.4 Pro, extended thinking | Gets most items right. Cites sources. No chart on first shot. ~$1/run. |

A dollar per run sounds cheap until you write down what the firm actually
wants to do with it. Fifteen hundred projects a year, twenty years of
history, a backlog of hypothetical backtests: *what if we'd switched lumber
suppliers in 2012? What happens to margins if this supply chain constraint
materializes?* One backtest is a Monte Carlo simulation over tens of
thousands of projects. At a dollar a run, that's the budget for a small
team.

So: same task, but on a 3B Granite model running on a laptop. About a
hundred lines of code. No API keys, no egress, no per-token billing. Here's
how.

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

def _bom_entry_is_well_formed(entry: BOMEntry) -> bool:
    """Quantity is either an integer or the string 'allowance'."""
    try:
        int(entry.quantity)
        return True
    except ValueError:
        return entry.quantity.lower() == "allowance"
```

"Allowance" is a construction term of art: *buy some of these, here's twenty
bucks.* The Pydantic schema alone can't express that rule, so it goes into a
requirement with an explicit validator:

```python
m.instruct(
    "Reformat this table to have four columns: item, quantity, type, and notes.",
    grounding_context={"table": table.to_markdown()},
    requirements=[
        req(
            "Quantity should only contain an integer or Allowance",
            validation_fn=simple_validate(_bom_entries_are_well_formed),
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
independent of the others. `ainstruct` returns a thunk (a deferred
computation) so the extraction can fan out:

```python
from mellea.core import ModelOutputThunk

async def extract_bom(doc: RichDocument):
    bom_routines = []
    for table in doc.get_tables():
        if is_material_list(m, table_markdown=table.to_markdown()) == "yes":
            bom_routines.append(m.ainstruct(..., format=BOM))

    bom_thunks: list[ModelOutputThunk] = [await r for r in bom_routines]
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
handles PDF, DOCX, and XLSX uniformly:

```python
from mellea.stdlib.components.docs import Document
from mellea.stdlib.components.docs.richdocument import RichDocument

rd_doors = RichDocument.from_document_file("product_catalogs/door_product_catalog.pdf")
doors_doc = Document(text=rd_doors.to_markdown())

rd_windows = RichDocument.from_document_file("product_catalogs/north_ridge_windows.docx")
windows_doc = Document(text=rd_windows.to_markdown())

rd_lumber = RichDocument.from_document_file("product_catalogs/cone_mountain_lumber_catalog.xlsx")
lumber_doc = Document(text=rd_lumber.to_markdown())
```

Each becomes a plain `Document`, the input format Mellea's RAG intrinsics
expect.

## Step 3: Match BOM Entries to Prices with RAG Intrinsics

This is the step that earns the pipeline. Matching "36x80 Aurora half-moon
entry door" from a bill of materials to the right line in a product catalog
is fuzzy by nature. Rather than ask a general-purpose model to "find the
right price and tell me if you're sure," Mellea exposes purpose-built
adapters from the [IBM Granite RAG intrinsics
library](https://huggingface.co/ibm-granite) for exactly this kind of check:
`check_context_relevance`, `check_answerability`, and `find_citations`.

The intrinsics run on the HuggingFace backend, so start a second session:

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
    question="What is the price of the 36x80 Aurora half-moon entry door?",
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

What makes these intrinsics worth reaching for, as opposed to asking a
general model "is this document relevant, 1-10," is that the outputs are
**calibrated**. Pass the doors catalog and a doors question, you get a
high-confidence "answerable." Pass the lumber catalog with the same doors
question, and context relevance drops to around 0.5 (a pricing document
about construction, but not the right one) while answerability correctly
collapses to "unanswerable." Frontier model logits don't give you this.
Nothing in a general-purpose model's training signal makes the raw
next-token probabilities mean "calibrated confidence that this document
answers this question." Intrinsics are trained to make them mean that.

### Why intrinsics are cheap to compose

A realistic pipeline wants to run multiple checks on the same
(question, document) pair: is it relevant, is it answerable, where are the
citations? Naively, three adapters means three full model runs, each
re-prefilling the same long document prompt.

Granite intrinsics ship as **ALoRA** adapters, not plain LoRAs. The
difference is narrow but matters here: an ALoRA reuses the base model's
prefill. The expensive pass over the document happens once against the
base weights, and each adapter only kicks in during token generation. Three
sequential checks on the same document share one prefill. You get
modularity without paying 3× the compute.

### The pricing loop

The pricing logic walks every BOM item, picks the right catalog by category,
and asks the answerability intrinsic whether a price is extractable before it
ever asks the model to produce one:

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

def get_prices(m: mellea.MelleaSession, bom: BOM) -> list[BomEntryWithPrice]:
    prices: list[BomEntryWithPrice] = []
    for entry in bom.items:
        catalog = {"windows": windows_doc, "doors": doors_doc}.get(entry.category)
        # Lumber is skipped in the tutorial to keep Colab T4 runtime reasonable.

        if catalog:
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
                continue

        # Item not covered by a catalog — record it with unknown price.
        prices.append(BomEntryWithPrice(
            item=entry.item, quantity=entry.quantity, notes=entry.notes,
            category=entry.category, unit_price=None, total_price=None,
        ))
    return prices
```

Two things matter about this loop. First, `verdict == "answerable"` is a
gate: items the intrinsic can't confidently answer don't get a hallucinated
price, they get `total_price=None` and flow through to the report as
"unknown." Failure modes are explicit. Second, the model is only ever asked
to extract a number from a document where a calibrated adapter has already
confirmed the number is there. The hard problem, *is this the right
document at all?*, was solved by a specialized adapter rather than by a
general prompt.

For extra belt-and-suspenders, `find_citations` will highlight the exact
span of the source document that produced each answer, so spot-checks
become a substring lookup rather than a re-read:

```python
from mellea.stdlib.components.intrinsic.rag import find_citations

citations = find_citations(
    response=price_response.value,
    documents=[doors_doc],
    context=ctx,
    backend=m_hf.backend,
)
```

## Step 4: Generate the Report

The final step is two instructions: make the chart, then write the report
around it.

Pricing runs on a 3B Granite model, but chart-drawing code is a different
beast. Tool-calling and matplotlib benefit from a model with more muscle,
so for this step switch sessions to GPT-OSS 20B running locally on Ollama:

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
open("/tmp/report.html", "w").write(report.value)
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

Step back from the construction example. What just happened is the general
shape of the trade.

![A small model, harnessed](/images/small-models-rock/harnessed.png)

A task that one-shot required a frontier model and a dollar per run now
runs on a 3B open-weight model on a laptop. Pricing extraction on Granite 4
Micro 3B via HuggingFace, chart and report generation on GPT-OSS 20B via
Ollama, no API keys, no egress, no per-token billing. Items the pipeline
can't confidently price show up as `unknown` in the output rather than as
hallucinated numbers, so the report is auditable instead of something a
human has to re-check line by line.

The construction case isn't a one-off. The same three-pattern approach
generalizes. On agent benchmarks the Mellea team has run (a DB2 database
agent and a compliance agent), rewriting large prompt-based systems as
Mellea programs moves a Llama 70B setup from ~80% task completion to ~90%,
and lets a Granite 8B model match or beat a Llama 70B baseline. Those last
ten points of task completion are the ones you usually can't buy without
going up a model tier. The harness buys them instead.

The practical upshot: "small model or frontier model" is less a question
about raw capability and more a question about whether the harness around
it is doing its share. If you're paying frontier-model prices for a task
that decomposes cleanly, there's a good chance you're paying for
engineering you haven't done yet.

## Trade-offs

Latency per run is higher than a single API call. Async execution hides
most of the fan-out, but a pipeline will always be slower per-run than one
prompt. The win is in cost, privacy, and auditability, not in tail latency.

Decomposition takes engineering effort. It's ordinary software work, and
the upfront cost is real. It pays back once the task is recurring, privacy
matters, or cost compounds at scale.

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
- [Granite RAG intrinsics on Hugging Face](https://huggingface.co/ibm-granite)
- [Construction tutorial notebook](https://github.com/generative-computing/mellea-tutorials/blob/main/notebooks/atai_2026/tutorial.ipynb)
