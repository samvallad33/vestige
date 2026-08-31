# Vestige v2.0 "Cognitive Leap" — MCP Dev Summit NYC Demo Script

**Event:** MCP Dev Summit NYC, April 1-3, 2026
**Presenter:** Sam Valladares
**Project:** Vestige — The cognitive engine that gives AI a brain.
**Tagline:** 130 years of memory research. One Rust binary. Zero cloud.

---

## Pre-Demo Checklist (Do This Before the Conference)

### Hardware
- [ ] MacBook charged, charger packed
- [ ] USB-C to HDMI adapter tested with venue projector (ask AV team for resolution)
- [ ] Phone hotspot configured as backup (embedding model already cached = no network needed)

### Software
- [ ] Vestige v2.0 binary installed: `vestige-mcp --version` shows `2.0.0`
- [ ] Claude Code installed and authenticated
- [ ] Terminal font size: 18pt minimum (audience readability)
- [ ] Browser zoom: 150% for dashboard views

### Pre-Load for Offline Demo
```bash
# 1. Ensure embedding model is cached (130MB, downloads on first use)
#    Run any search to trigger download:
vestige health

# 2. Verify cache exists:
ls ~/.fastembed_cache/
# Should show: nomic-ai--nomic-embed-text-v1.5/

# 3. Pre-load ~20 diverse memories so the graph looks alive:
vestige ingest "TypeScript is my preferred language for frontend development" --tags preference,typescript
vestige ingest "React Server Components eliminate client-side data fetching waterfalls" --tags pattern,react
vestige ingest "Decided to use Axum over Actix-web for Vestige's HTTP layer because of tower middleware ecosystem" --tags decision,architecture
vestige ingest "BUG FIX: SQLite WAL mode requires separate reader/writer connections for concurrent access" --tags bug-fix,sqlite
vestige ingest "FSRS-6 uses 21 parameters trained on 700M+ Anki reviews, achieving 30% better efficiency than SM-2" --tags fsrs,science
vestige ingest "Prediction Error Gating: similarity > 0.92 = reinforce, > 0.75 = update, < 0.75 = create new" --tags pe-gating,science
vestige ingest "Synaptic Tagging (Frey & Morris 1997): memories become important retroactively when a significant event occurs within 9 hours" --tags science,synaptic-tagging
vestige ingest "Three.js InstancedMesh renders 1000+ nodes at 60fps using GPU instancing" --tags pattern,threejs
vestige ingest "MCP protocol uses JSON-RPC 2.0 over stdio — no HTTP overhead, native tool integration" --tags mcp,architecture
vestige ingest "Bjork dual-strength model: storage strength never decreases, retrieval strength decays with time" --tags science,bjork
vestige ingest "HyDE query expansion classifies intent into 6 types and generates 3-5 hypothetical document variants" --tags hyde,search
vestige ingest "Ebbinghaus forgetting curve: R = e^(-t/S) where R=retrievability, t=time, S=stability" --tags science,ebbinghaus
vestige ingest "USearch HNSW index is 20x faster than FAISS for nearest neighbor search" --tags performance,search
vestige ingest "Reconsolidation (Nader 2000): retrieved memories enter a labile state for 24-48 hours where they can be modified" --tags science,reconsolidation
vestige ingest "Anderson 1994 retrieval-induced forgetting: retrieving one memory suppresses competing memories" --tags science,competition
vestige ingest "Einstein & McDaniel 1990 prospective memory: remember to do X when Y happens, with time/context/event triggers" --tags science,prospective-memory
vestige ingest "Vestige search pipeline: overfetch -> rerank -> temporal boost -> accessibility filter -> context match -> competition -> spreading activation" --tags architecture,search-pipeline
vestige ingest "Vestige has 734 tests, 77,840 lines of Rust, 29 cognitive modules, and ships as a 22MB binary" --tags vestige,stats
vestige ingest "The difference between Vestige and every other AI memory tool: we implemented the actual neuroscience, not just a vector database with timestamps" --tags vestige,philosophy
vestige ingest "Spreading activation: when you search for one thing, related memories light up automatically, like how thinking of 'doctor' primes 'nurse'" --tags science,spreading-activation

# 4. Run a dream cycle to create connections between memories:
#    (Do this through Claude Code so the CognitiveEngine processes it)
#    In Claude Code, say: "Dream about my recent memories"

# 5. Verify dashboard is running:
open http://localhost:3927/dashboard
# Should see 3D graph with ~20 nodes, connections from the dream

# 6. Test the full demo flow once (time it):
#    - Open terminal, open browser side by side
#    - Run through the 3-minute version
#    - Target: under 3 minutes with natural pacing
```

### Browser Tabs to Pre-Open
1. `http://localhost:3927/dashboard` (3D Graph view)
2. `http://localhost:3927/dashboard/feed` (Real-time event feed)
3. `http://localhost:3927/dashboard/stats` (Stats with retention histogram)
4. GitHub repo: `https://github.com/samvallad33/vestige`

### Terminal Setup
- Split terminal: left pane for Claude Code, right pane for commands
- Dark background, high contrast
- Cursor blink off (less distracting on projector)

---

## VERSION 1: 30-Second Elevator Pitch

**Use this:** Hallway conversations, after-party, meeting someone at the coffee line, anyone who asks "what are you working on?"

### The Script

> Your AI forgets everything between sessions. Every conversation starts from zero.
>
> I built Vestige — a single Rust binary that gives AI persistent memory based on real cognitive science. Not a vector database with a timestamp column. Actual FSRS-6 spaced repetition trained on 700 million reviews. Prediction error gating. Synaptic tagging. Memory dreaming. The same algorithms your brain uses.
>
> It runs as an MCP server. One command to install, one command to connect. Your AI remembers your preferences, your decisions, your bug fixes. And memories decay on the Ebbinghaus curve unless they're used — just like yours do.
>
> Twenty-two megabyte binary. Seven hundred thirty-four tests. Zero cloud dependencies. I'm at the summit if you want to see the 3D brain visualization.

### Key Points to Hit
- "Real neuroscience, not just embeddings" — this is the differentiator
- "Single Rust binary" — simplicity, performance
- "MCP server" — relevant to this audience specifically
- "Zero cloud" — privacy, local-first resonates
- End with an invitation to see the dashboard — creates a follow-up

### If They Ask One Follow-Up Question
**"How is this different from Mem0?"**
> Mem0 is a cloud memory API. Great product, well-funded. But it's fundamentally a vector store with categories. Vestige implements the actual cognitive science — memories decay on the Ebbinghaus curve, strengthen after explicit positive feedback, get consolidated in dream cycles, compete for activation. It's the difference between a filing cabinet and a brain.

**"What's the MCP integration like?"**
> One command: `claude mcp add vestige vestige-mcp -s user`. That's it. Twenty-one tools, but they're organized into five subsystems that Claude uses automatically. You don't even think about it — your AI just starts remembering.

**"Is it open source?"**
> AGPL-3.0. Fully open. The neuroscience is the moat, not the code.

---

## VERSION 2: 3-Minute Demo — "Watch Me Think"

**Use this:** Lightning talk slot, booth demo, small group gathered around your laptop.

**Tone:** Fast. Visual. Punchy. Let the 3D graph do the talking.

**Setup:** Terminal on left half of screen. Browser with dashboard on right half. Dashboard open to the Graph page.

### [0:00-0:20] The Hook

> *[Point at 3D graph on screen, nodes floating and pulsing]*
>
> This is a brain. Not a metaphor — an actual implementation of how memory works. Every node is a memory. Every connection was discovered by a dream cycle. That glow you see? That's retention decaying on the Ebbinghaus curve in real time.
>
> This is Vestige. Let me show you what happens when I talk to Claude.

### [0:20-0:50] The Live Memory

*[Switch to terminal with Claude Code]*

```
You: Remember that I'm presenting at MCP Dev Summit NYC and my talk is about cognitive memory systems
```

*[While Claude processes, switch to browser — point at the Feed tab]*

> Watch the feed. See that? "MemoryCreated" event just fired over WebSocket. And if you look at the graph...
>
> *[Switch to Graph tab — a new node appears with a burst animation]*
>
> There. A new neuron, literally being born in real time.

### [0:50-1:30] The Search

*[Back to terminal]*

```
You: What do you know about how I'm using memory science in my work?
```

> Now watch what happens during a search.
>
> *[Switch to Feed — show SearchPerformed event]*
>
> Seven-stage pipeline just fired. It overfetched 3x results, ran them through a cross-encoder reranker, applied temporal boost, checked FSRS retention, matched context using Tulving's encoding specificity principle, ran competition — Anderson's retrieval-induced forgetting — and spread activation to related memories.
>
> *[Point at Claude's response showing recalled context]*
>
> And it just pulled back exactly the right context. Not because of keyword matching — because of cognitive science.

### [1:30-2:15] The Dream

*[Back to terminal]*

```
You: Dream about my recent memories
```

> Now the wild part. This triggers a dream cycle — inspired by how your hippocampus replays memories during sleep to find hidden connections.
>
> *[Switch to Graph — show dream mode: purple ambient, nodes pulsing sequentially]*
>
> Watch the nodes light up one by one. It's replaying memories, looking for patterns it missed. See those golden lines appearing? Those are connections it just discovered between memories that were stored at completely different times.
>
> *[Switch to Stats tab, point at retention histogram]*
>
> And here's the retention distribution. FSRS-6 — trained on 700 million Anki reviews. Every memory has a predicted decay curve. Memories that aren't accessed fade. Memories that get used get stronger. Exactly like your brain.

### [2:15-2:50] The Punchline

> *[Switch to terminal, run:]*

```bash
vestige-mcp --version
# → vestige-mcp 2.0.0
```

```bash
wc -l $(find /path/to/vestige/crates -name "*.rs") | tail -1
# → 77,840 total
```

> Seventy-eight thousand lines of Rust. Seven hundred thirty-four tests. Twenty-two megabyte binary. Ships with the dashboard embedded. Install is one npm command:

```bash
npm install -g vestige-mcp-server
claude mcp add vestige vestige-mcp -s user
```

> Two lines. Your AI now has a brain.

### [2:50-3:00] Close

> Vestige v2.0, "Cognitive Leap." Open source, AGPL-3.0. The repo is `samvallad33/vestige`. Come talk to me if you want to see the neuroscience under the hood.

---

## VERSION 3: 10-Minute Deep Dive — Full Walkthrough

**Use this:** Breakout session, workshop demo, recorded talk, anyone who gives you a stage.

**Tone:** Authoritative. Technical depth but accessible. Build from simple to mind-blowing.

**Setup:** Terminal fullscreen. Browser in separate space (swipe to switch). Have the GitHub repo open as a backup.

---

### ACT 1: The Problem [0:00-1:30]

> I want to start with a question. How many of you use Claude, or GPT, or Cursor every day?
>
> *[Hands go up]*
>
> And how many of you have had this experience: you spent two hours debugging a problem with your AI, you finally fixed it, and a week later you hit the exact same bug and your AI has absolutely no memory of the solution?
>
> *[Nods]*
>
> That's because every major AI assistant has the memory of a goldfish. Every session starts from absolute zero. Claude literally tells you this: "I don't have the ability to remember previous conversations."
>
> Now here's what's strange. We've known how to build memory systems for over a century. Hermann Ebbinghaus published the forgetting curve in 1885. Bjork and Bjork formalized the dual-strength model in 1992. FSRS-6 — the state of the art in spaced repetition — was trained on 700 million reviews and published in 2024. The science exists.
>
> Nobody implemented it for AI. So I did.

### ACT 2: Install and First Memory [1:30-3:30]

> Let me show you how fast this is. I'm starting from scratch.

```bash
# Install (macOS Apple Silicon)
npm install -g vestige-mcp-server
```

> Three binaries. The MCP server, the CLI admin tool, and a restore utility. Twenty-two megabytes total. No Docker. No Python. No node_modules. No cloud API key.

```bash
# Connect to Claude Code
claude mcp add vestige vestige-mcp -s user
```

> That's it. One command connects Vestige to Claude as a user-scoped MCP server. Let me restart Claude Code and show you what happens.

*[Open Claude Code]*

```
You: Remember that I prefer Rust over Go for systems programming, and TypeScript for frontend work.
```

> Watch what Claude does. It calls `smart_ingest` — Vestige's primary storage tool. But this isn't just shoving text into a database. Let me walk you through what just happened under the hood:
>
> 1. **Prediction Error Gating** checked if this memory already exists. It compared the new content against all stored memories using embedding similarity. Since it's novel — similarity below 0.75 — it creates a new memory.
> 2. **Importance scoring** ran four channels: novelty (is this new?), arousal (is this emotionally significant?), reward (will this be useful?), attention (is the user focused?). User preferences score high on reward.
> 3. **Intent detection** classified this as a "preference" statement.
> 4. **Synaptic tagging** marked this memory for potential retroactive strengthening if something related happens in the next 9 hours.
> 5. **Hippocampal indexing** created a fast-lookup pointer.
>
> All of that in under 50 milliseconds. In a single Rust binary. No API calls.

*[New Claude Code session]*

```
You: What programming languages do I prefer?
```

> New session. Clean context. But Vestige remembers.
>
> *[Show Claude responding with the saved preference]*
>
> And this search just ran a 7-stage cognitive pipeline. Let me show you what's happening visually.

### ACT 3: The Dashboard [3:30-5:30]

> *[Switch to browser: `localhost:3927/dashboard`]*
>
> This is the Vestige dashboard. SvelteKit 2, Three.js, WebSocket connection to the running MCP server.

*[Graph page — nodes floating in 3D space with bloom post-processing]*

> Every node is a memory. The brightness represents retention strength — how accessible this memory is right now. That's driven by FSRS-6, the same algorithm that powers modern Anki. Brighter means recently accessed or frequently used. Dimmer means it's fading.
>
> The connections between nodes? Those were discovered during dream cycles — offline consolidation where Vestige replays memories and finds hidden relationships, just like your hippocampus does during sleep.
>
> I can click any node to see its details — content, type, tags, retention strength, when it was created, when it was last accessed, its predicted decay curve.

*[Click a node, show detail panel with retention curve]*

> See this curve? That's `R(t) = (1 + FACTOR * t / S)^(-w20)` — the FSRS-6 forgetting curve. Right now this memory has 94% retention. In 7 days without access, it drops to 71%. In 30 days, 43%. But every time it's retrieved, the stability increases and the curve flattens. The more you use a memory, the harder it is to forget. Exactly like your brain.

*[Switch to Feed page]*

> This is the real-time event feed. Every cognitive operation — memory creation, search, dreaming, consolidation — fires a WebSocket event. Let me show you.

*[Switch back to terminal, run a search in Claude Code]*

```
You: How does FSRS-6 work?
```

*[Switch to Feed — show SearchPerformed event with details]*

> There — `SearchPerformed`. Query: "How does FSRS-6 work?" Results: 3 memories returned. And on the graph, watch the nodes involved pulse.
>
> *[Switch to Graph — show nodes pulsing from the search]*
>
> The blue pulse you see is spreading activation. When you search for FSRS, it doesn't just find memories about FSRS — it activates related memories about spaced repetition, Ebbinghaus, Bjork. Like how thinking about "doctor" primes "nurse" in your mind.

### ACT 4: The Dream [5:30-7:30]

> Now let me show you the feature I'm most proud of. Memory dreaming.
>
> Your brain doesn't just store memories and retrieve them. During sleep, your hippocampus replays the day's experiences, compresses them, finds connections you missed while awake, and consolidates the important ones into long-term storage. This is the science of memory consolidation — Diekelmann and Born, 2010.
>
> Vestige does the same thing.

*[In Claude Code:]*

```
You: Dream about my recent memories
```

*[Switch to Dashboard Graph — show dream mode activate]*

> Watch this. Purple ambient wash — we're entering dream mode. The graph slows down. And now watch the nodes light up one at a time.
>
> *[Point as nodes pulse sequentially]*
>
> It's replaying each memory, computing similarity to every other memory, looking for connections that weren't obvious at ingest time. See that golden line that just appeared? It just discovered that two memories stored days apart are semantically related.

*[Switch to Feed — show DreamStarted, then DreamCompleted events]*

> The dream cycle produces insights — natural language descriptions of the connections it found. And here's the critical part: the memories that get replayed during dreaming are strengthened. Their FSRS stability increases. Memories that aren't replayed continue to decay. Over time, the system naturally retains what matters and forgets what doesn't.
>
> This is not a cron job. This is not garbage collection. This is the actual computational equivalent of memory consolidation during sleep.

### ACT 5: HyDE and the Search Pipeline [7:30-9:00]

> One more thing I want to show you. The search isn't just keyword matching plus cosine similarity.
>
> Vestige v2.0 added HyDE — Hypothetical Document Embeddings. When you search for something conceptual, the system first classifies your intent: are you asking a definition question? A how-to? Reasoning about a problem? Looking up a specific fact?
>
> Based on that classification, it generates 3 to 5 hypothetical documents — what a perfect answer might look like — and creates a centroid embedding from all of them. That centroid is what gets compared against your stored memories.
>
> The result: dramatically better recall on conceptual queries. If you search "how does memory decay work," you don't just get memories that contain those words. You get memories about Ebbinghaus, FSRS, retention curves, Bjork's dual-strength model — because the hypothetical documents capture the concept, not just the keywords.
>
> This runs on top of the 7-stage pipeline:
>
> 1. Overfetch 3x results from hybrid search (BM25 keyword + semantic embedding)
> 2. Cross-encoder reranker re-scores by deep relevance
> 3. Temporal boost for recent memories
> 4. FSRS-6 retention filter — memories below threshold are inaccessible
> 5. Context matching — Tulving 1973, encoding specificity
> 6. Competition — Anderson 1994, retrieval-induced forgetting
> 7. Spreading activation — activate related memories, update predictive model
>
> And critically: search does not mistake relevance for correctness. Retrieval is audit-only; once a memory proves useful, an explicit promote applies the Testing Effect without pinning stale results.

### ACT 6: The Numbers and the Close [9:00-10:00]

> Let me leave you with the numbers.

*[Terminal:]*

```bash
vestige-mcp --version
# → vestige-mcp 2.0.0

# Stats
# 77,840 lines of Rust
# 734 tests, zero failures
# 29 cognitive modules
# 22MB release binary with embedded dashboard
# 21 MCP tools across 5 subsystems
# 12 published neuroscience principles implemented
# <50ms typical ingest latency
# <300ns cosine similarity (benchmarked with Criterion)
# Zero cloud dependencies
# Zero API keys required
# One command to install
```

> This is what I've been building for the past three months. I'm one person, I'm twenty-one years old, and I believe this is how AI memory should work — grounded in real science, running locally, open source.
>
> Vestige v2.0, "Cognitive Leap." The repo is `github.com/samvallad33/vestige`. The dashboard is running at `localhost:3927`. I'll be around all three days — come find me if you want to talk about FSRS, or synaptic tagging, or why I think every AI assistant on the planet should have a forgetting curve.
>
> Thank you.

---

## Anticipated Audience Questions and Answers

### Technical Questions

**Q: How does this compare to RAG? Isn't this just RAG with extra steps?**
> RAG is retrieval-augmented generation — you search a corpus and inject results into the prompt. Vestige does that, but with a cognitive layer on top. RAG doesn't have retention decay. RAG doesn't have memory consolidation. RAG doesn't have prediction error gating to prevent duplicates. RAG doesn't suppress competing memories on retrieval. Vestige is to RAG what human memory is to a filing cabinet — the retrieval mechanism is similar, but the memory lifecycle is completely different.

**Q: What embedding model do you use?**
> Nomic Embed Text v1.5 by default — 768 dimensions truncated to 256 via Matryoshka representation learning. All local via ONNX through fastembed. v2.0 also supports Nomic v2 MoE (475M params, 8 experts) as an opt-in feature. The reranker is Jina v1 Turbo, with Qwen3-Reranker-0.6B available as opt-in.

**Q: What's the storage backend?**
> SQLite with WAL mode. FTS5 for keyword search with Porter stemming. USearch HNSW for vector search — 20x faster than FAISS. Separate reader/writer connections for concurrent access. Single file database. I8 vector quantization for 2x storage savings with under 1% recall loss.

**Q: How does FSRS-6 actually work? What are the 21 parameters?**
> FSRS models memory as a power-law forgetting curve: `R(t) = (1 + FACTOR * t / S)^(-w20)` where S is stability and w20 is the decay parameter. The 21 parameters were trained on 700 million Anki reviews using machine learning. They encode how difficulty changes with repeated reviews, how stability grows based on review quality (Again/Hard/Good/Easy), and how same-day reviews affect long-term retention. It's 30% more efficient than SM-2, which is what Anki has used for decades.

**Q: Does it work with Claude Desktop? Other AI clients?**
> Yes. It speaks MCP — the Model Context Protocol. One config change and it works with Claude Desktop, Cursor, VS Code Copilot, JetBrains, Windsurf, Xcode 26.3. Anything that speaks MCP.

**Q: What about multi-user or team memory?**
> The current Pro plan is more pragmatic: prove portable sync/storage first, then ship Solo and Team workflows around managed sync, backups, onboarding, and support. The open-source core stays local-first; paid team features should stay in the separate Pro/commercial boundary.

**Q: How does Prediction Error Gating prevent duplicate memories?**
> When you ingest a new memory, it computes embedding similarity against all existing memories. If similarity is above 0.92, it reinforces the existing memory (bumps FSRS stability). Between 0.75 and 0.92, it updates/merges. Below 0.75, it creates a new memory. The thresholds come from computational neuroscience research on prediction error signals — the brain stores what's surprising, reinforces what's familiar, and updates what's partially known. Same principle.

**Q: What's the performance like at scale? How many memories can it handle?**
> I'm currently running with 700+ memories and everything is under 50ms. HNSW scales logarithmically, so 10x the memories doesn't mean 10x the latency. SQLite handles millions of rows. The practical limit is probably in the hundreds of thousands before you'd want to shard — but for personal AI memory, that's years of heavy use.

### Business / Community Questions

**Q: Why AGPL and not MIT?**
> Because I don't want AWS or Google hosting Vestige as a service without contributing back. AGPL means if you serve it over a network, you must open-source your modifications. Local use is completely free and unrestricted. Cloud and team features are proprietary — the MongoDB/HashiCorp playbook.

**Q: How do I contribute?**
> GitHub: `samvallad33/vestige`. The codebase is well-tested — 734 tests, zero warnings. Good first issues are labeled. The science is documented in `docs/SCIENCE.md`. PRs welcome, especially for new cognitive modules.

**Q: Are you funded? Is this a company?**
> Not yet. This is a solo project. I built it because I believe AI memory should work like human memory — not because a VC told me to. If the right opportunity comes along, I'm open to it. But the open source core isn't going anywhere.

### Skeptical Questions

**Q: Isn't this over-engineered? Do you really need 29 cognitive modules?**
> Fair question. You could build a memory system with just embeddings and timestamps, and it would work for simple cases. But it would miss things. It wouldn't know that a memory stored last week is related to one stored last month unless you searched for both. It wouldn't automatically forget outdated information. It wouldn't retroactively strengthen a memory when something important happens later. Each module solves a specific problem that real users hit. The 29 modules aren't bloat — they're the necessary completeness of a system that actually models cognition.

**Q: How do I know the neuroscience is real and not just marketing?**
> Every principle I've implemented has a citation. Ebbinghaus 1885. Bjork and Bjork 1992. Frey and Morris 1997. Anderson 1994. Tulving 1973. Einstein and McDaniel 1990. Nader 2000. Diekelmann and Born 2010. FSRS was published with full methodology and trained on public Anki data. I didn't make any of this up — I translated peer-reviewed cognitive science into Rust.

**Q: Claude Code just added built-in memory. Doesn't that make Vestige obsolete?**
> Claude's built-in memory is a flat text file. No spaced repetition, no decay, no dreaming, no cognitive search pipeline, no visualization. It's a good first step — it validates that persistent memory matters. But it's the difference between a notepad and a brain. Vestige doesn't compete with Claude's memory — it replaces and extends it.

---

## Recovery Plans

### If the Dashboard Doesn't Load
- Vestige-mcp must be running (it auto-starts the dashboard on port 3927)
- Fall back to terminal-only demo: `vestige health` and `vestige stats` show system state
- Say: "The dashboard is SvelteKit embedded in the binary — it starts automatically. Let me show you the data from the terminal instead."

### If the Embedding Model Downloads Mid-Demo
- This only happens on first-ever use. The pre-demo checklist prevents this.
- If it happens: "The embedding model is 130MB and downloads once on first use. After that, Vestige is fully offline. Let me switch to the dashboard while it downloads."

### If Claude Code Takes Too Long to Respond
- Have a second terminal tab with pre-typed commands ready
- Switch to showing the dashboard and explaining the architecture
- Say: "While Claude processes that, let me show you what's happening in the visualization."

### If Someone Asks Something You Don't Know
- "That's a great question and I honestly don't have a good answer for it yet. Find me after the talk and let's dig into it."
- Never bluff. The audience is technical. Authenticity wins.

---

## Stage Presence Notes

- **Start from the dashboard.** The 3D graph is the hook. It's visual, it's unusual, it makes people lean in.
- **Don't rush the dream sequence.** The purple wash and sequential node pulses are the most visually impressive moment. Let it breathe for 3-4 seconds.
- **Say the scientists' names.** "Ebbinghaus," "Bjork," "Frey and Morris" — this signals that you've done the reading. The MCP Dev Summit audience respects depth.
- **Make eye contact during the punchline.** "One command. Your AI now has a brain." Look at the audience, not the screen.
- **Own your age.** Twenty-one, solo developer, zero funding. This is an asset, not a liability. You built something that the well-funded competitors haven't.
- **The dashboard is your co-presenter.** Every time Claude does something, the dashboard should be showing the corresponding event. Practice the terminal-to-browser switch until it's seamless.
- **Don't apologize.** Not for bugs, not for the AGPL, not for being solo. Confident but not arrogant. The work speaks.
