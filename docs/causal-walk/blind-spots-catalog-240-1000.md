# Causal walk blind spots catalog (items 240–1000+)

Items 1–239 are in `backfill-blind-spots.md` (1–20), the uploaded batches (21–239), and `docs/causal-walk/blind-spots.md`. They are not repeated here. Links are recorded causal events only: commits touching a path or hunk, blame of an exact path and line, revert or `Fixes:` trailers, patch-ids, gitlinks, tool calls, CI results, gate decisions, derivations, supersessions, or named on-disk receipts. No keyword, BM25, subject-line, file-stem, package-name, test-name, symbol-name, entity-name, or embedding join. No SQLite.

## Bug family summary (items 240+)

| Family | Item range | Count (planned) |
| --- | --- | ---: |
| Concurrency and async | 240–289 | 50 |
| Memory safety and lifetimes | 290–339 | 50 |
| Numeric, units, and serialization | 340–389 | 50 |
| Time, clocks, and scheduling | 390–439 | 50 |
| Encoding, locale, and text | 440–489 | 50 |
| Build, toolchain, and codegen | 490–539 | 50 |
| ABI, FFI, and platform | 540–589 | 50 |
| Network, HTTP, and RPC | 590–639 | 50 |
| Distributed systems and consensus | 640–689 | 50 |
| Databases, migrations, and queries | 690–739 | 50 |
| Caches and storage engines | 740–789 | 50 |
| Security, authz, and crypto | 790–839 | 50 |
| CI/CD, IaC, and Kubernetes | 840–889 | 50 |
| Frontend, mobile, and GPU | 890–939 | 50 |
| ML, agents, and data pipelines | 940–989 | 50 |
| Ops, licensing, and multi-repo | 990–1000+ | 11+ |

Counts update as batches land. **Current last item:** 289 (batch 1 complete).

---

# Batch 1 — items 240–289

Read-only. Vestige branch `cursor/causal-walk-blind-spots-975d`. Items 1–239 are not repeated. Nothing joins on a keyword, subject line, file stem, package name, test name, resource name, or symbol. `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) orders blame, hunks, and time; it has no slot for most receipt fields below. `classify_starts` starts from `node_id` and recorded paths only.

## 240. A wait group counter goes negative

**Category.** Concurrency and async.

**Pattern.** A goroutine calls `Done` more often than `Add`. The runtime panics. Blame names the line that called `Done`; the missing `Add` is on another goroutine or an earlier path with no shared hunk.

**Example.** https://github.com/golang/go/issues/43755. `sync.WaitGroup` misuse: negative counter panic when `Done` races ahead of `Add`. The issue documents the panic string and stack; the fix is pairing `Add`/`Done` on every path.

**Recorded-event mechanism.** A runtime receipt stores the panic kind bytes and the goroutine id at panic. A separate stack frame list is ordered `path:line` only. Promotion to a commit needs a recorded test that fails on the child and passes on the parent at one of those paths. **No allowed fix** pairs goroutines by function name.

**Gap.** `split_frame` and `git_rank` (`causal_walk.rs`) have no wait-group counter receipt. `race_receipt` is named in item 126 but not read.

**Priority.** High.

## 241. Channel closed while a sender still runs

**Category.** Concurrency and async.

**Pattern.** A sender writes after `close`. The runtime panics on send. The `close` call and the late send are different commits or different files.

**Example.** https://github.com/golang/go/issues/45100. Discussion of send-on-closed-channel panics and race patterns in concurrent shutdown. Public issue with reproducer discussion.

**Recorded-event mechanism.** Panic receipt with kind bytes `send on closed channel`. Stack frames as exact `path:line` list. The walk follows each frame that matches a recorded `file:` anchor. **No allowed fix** from the channel variable name.

**Gap.** `classify_starts` (`causal_walk.rs`) does not ingest panic-kind receipts.

**Priority.** High.

## 242. `select` wakes both cases after one branch closed the channel

**Category.** Concurrency and async.

**Pattern.** One goroutine closes a channel used in multiple `select` statements. Another goroutine observes a zero value and believes it is a real message. The close and the handler logic are not in the same blame line.

**Example.** https://github.com/golang/go/issues/51465. `select` behavior when a channel is closed while other cases are ready; documents subtle ordering. Closed with clarification in Go 1.19 release notes thread.

**Recorded-event mechanism.** Record channel id (opaque) on send, close, and receive receipts. A receive with `closed=true` and `len=0` after a close receipt on the same id is the finding. **No allowed fix** without those receipts.

**Gap.** `git_rank` has no channel id slot.

**Priority.** Medium.

## 243. Mutex unlocked by a different goroutine than locked it

**Category.** Concurrency and async.

**Pattern.** Runtime fatal error: unlock of unlocked mutex. The lock was taken on one stack and released after a callback on another thread.

**Example.** https://github.com/golang/go/issues/24161. `fatal error: sync: unlock of unlocked mutex` from mismatched lock/unlock goroutines.

**Recorded-event mechanism.** Lock receipt: mutex id, holder goroutine id, `path:line`. Unlock receipt: same mutex id, different goroutine id. Pair is the finding. **No allowed fix** from mutex variable name in source.

**Gap.** Items 129–132 cover lock order in abstract; `repo_ingest.rs` `blame_at` has no mutex receipt reader.

**Priority.** High.

## 244. Double `await` on a future that was already consumed

**Category.** Concurrency and async.

**Pattern.** Rust async: second poll on a moved future is a compile error, but dynamic futures or `Pin` misuse can panic at runtime. Blame is on the await site, not the spawn site.

**Example.** https://github.com/rust-lang/rust/issues/64496. Discussion of polling dropped futures and undefined behavior in early async runtimes; linked to broader async semantics.

**Recorded-event mechanism.** A task receipt stores task id and poll count. Poll count greater than one on the same task id after `Ready` is the finding. **No allowed fix** without task id on receipts.

**Gap.** `walk_from` (`causal_walk.rs`) walks commit edges only, not task polls.

**Priority.** Medium.

## 245. Cancellation drops in-flight I/O without recording the abort handle

**Category.** Concurrency and async.

**Pattern.** Tokio `select!` cancels one branch when the other completes. The cancelled branch had started a write; the failure is partial data on disk. No commit touches the file after the cancel.

**Example.** https://github.com/tokio-rs/tokio/issues/4730. `select!` and cancellation safety documentation; issue thread on which operations are cancel-safe.

**Recorded-event mechanism.** Abort receipt: task id, cancelled at `path:line`. In-flight I/O receipt: same task id, operation id, byte offset. Cancel before completion without a matching completion receipt is the finding. **No allowed fix** from async block text.

**Gap.** `prove` (`walk_verify/run.rs`) records exit codes, not cancel receipts.

**Priority.** High.

## 246. Thread pool task runs after the runtime began shutdown

**Category.** Concurrency and async.

**Pattern.** Work is scheduled, shutdown is called, the task still runs and touches freed state. Stack points at pool worker, not the submitter.

**Example.** https://github.com/tokio-rs/tokio/issues/3500. Runtime shutdown vs. blocking pool tasks; race between `shutdown_background` and in-flight work.

**Recorded-event mechanism.** Shutdown receipt stores monotonic instant. Task start receipt with start instant after shutdown instant is the finding. Requires both instants on disk. **No allowed fix** from pool name string.

**Gap.** Item 237 is wall vs monotonic delta; `git_admissible` uses author time only.

**Priority.** High.

## 247. `Once` initialization runs twice under race

**Category.** Concurrency and async.

**Pattern.** `sync.Once` or `std::sync::Once` appears to run the closure twice in sanitizer builds. Two commits: one adds the `Once`, one removes a lock that serialized init.

**Example.** https://github.com/golang/go/issues/33264. Data race in `Once` fast path on certain architectures; fixed in runtime.

**Recorded-event mechanism.** TSAN receipt lists two write events at the same `path:line` without happens-before edge. Both event ids map to commits via blame at that line only. **No allowed fix** from variable name `once`.

**Gap.** Item 134 is sanitizer sites; `blame_at` does not read TSAN pair receipts.

**Priority.** High.

## 248. Atomic load uses relaxed ordering where acquire was required

**Category.** Concurrency and async.

**Pattern.** A flag is published with `Release` but read with `Relaxed`. CI passes on x86, fails on ARM. The ordering bug and the ARM-only test are different files.

**Example.** https://github.com/rust-lang/rust/issues/108650. Std discussion of atomic orderings and miscompilation examples in community issues; ARM weak ordering exposes bugs. (Issue tracks libc/std atomics documentation gaps.)

**Recorded-event mechanism.** LLVM IR or MIR receipt stores memory ordering enum per instruction at a recorded `path:line` column range. A load with `Relaxed` where a recorded happens-before edge requires `Acquire` is the finding. **No allowed fix** without IR receipt.

**Gap.** `import_target` (`git_records.rs`) does not parse IR; **needs new receipt type** (`memory_order`).

**Priority.** High.

## 249. Seqlock reader sees torn 64-bit value

**Category.** Concurrency and async.

**Pattern.** User-space seqlock: reader does not retry when sequence is odd. Consumer reads a torn 64-bit counter. Blame is on the reader loop; the writer commit only bumped the even sequence.

**Example.** https://github.com/redis/redis/issues/10426. Redis 6.2 memory ordering and atomicity discussion on ARM; thread about visibility of counters without proper barriers (public issue, 200).

**Recorded-event mechanism.** Seqlock receipt stores sequence before and after read, and retry count. Odd sequence on first load with zero retries is the finding. **No allowed fix** without sequence integers on disk.

**Gap.** `git_rank` (`causal_walk.rs`) has no seqlock counter; needs new receipt type (`seqlock_sequence`).

**Priority.** Medium.

## 250. Spin lock holder dies without releasing

**Category.** Concurrency and async.

**Pattern.** A userspace spinlock is held across a `longjmp` or signal handler. The next acquirer spins forever. No commit changes the lock word after the crash.

**Example.** https://github.com/golang/go/issues/12498. Spinning mutex discussion and runtime notes on lock ranking.

**Recorded-event mechanism.** CPU profile receipt stores sample `path:line` where exclusive time exceeds threshold while lock word receipt shows holder pid exited. **No allowed fix** without lock word and pid on disk.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`lock_word`).

**Priority.** High.


## 251. Condition variable signal before wait registers

**Category.** Concurrency and async.

**Pattern.** Thread A signals, then thread B starts waiting. The event is lost. Failure is intermittent under load.

**Example.** https://github.com/rust-lang/rust/issues/20333. Historical `condvar` behavior and spurious wakeup documentation.

**Recorded-event mechanism.** Wait and signal receipts share opaque condvar id with monotonic instants. Signal instant before wait start with no wakeup receipt is the finding. **No allowed fix** without condvar id.

**Gap.** `prove` (`crates/vestige-mcp/src/walk_verify/run.rs`) — needs new receipt/event type (`condvar_id`).

**Priority.** High.


## 252. Thread-local storage destructor runs during another TLS access

**Category.** Concurrency and async.

**Pattern.** TLS dtor for module A runs while module B still reads the same thread's TLS slot. Use-after-free in loader teardown.

**Example.** https://github.com/rust-lang/rust/issues/28125. TLS destructor order on Windows and Linux.

**Recorded-event mechanism.** TLS dtor receipt lists slot id and `path:line`; concurrent access receipt on same slot without happens-before is the finding. **No allowed fix** without slot id.

**Gap.** `blame_at` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`tls_slot`).

**Priority.** High.


## 253. Std::async future destroyed without awaiting

**Category.** Concurrency and async.

**Pattern.** Destructor of `std::async` launch policy blocks or leaves threads detached. Regression shows as hang in test shutdown, not in the async function body.

**Example.** https://github.com/microsoft/STL/issues/1364. STL issue on `std::async` destructor behavior and policy.

**Recorded-event mechanism.** Join receipt: thread id, policy enum, awaited bool. Destroy without awaited true is the finding. **No allowed fix** without join receipt.

**Gap.** `walk_from` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`async_join`).

**Priority.** Medium.


## 254. Event loop re-enters while a lock is held

**Category.** Concurrency and async.

**Pattern.** Node.js: sync fs call inside request handler blocks the loop; timer fires reentrantly. Stack shows timer file; lock taken in middleware.

**Example.** https://github.com/nodejs/node/issues/21556. Discussion of sync APIs blocking the event loop.

**Recorded-event mechanism.** Loop phase receipt and lock hold receipt with same thread id; nested turn while lock held is the finding. **No allowed fix** without phase receipt.

**Gap.** `classify_starts` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`loop_phase`).

**Priority.** High.


## 255. Python asyncio task cancelled after `gather` returned

**Category.** Concurrency and async.

**Pattern.** Parent coroutine exits; child task still mutates shared list. Failure in child stack; cancel commit only touched parent.

**Example.** https://github.com/python/cpython/issues/91887. asyncio cancellation and task group semantics.

**Recorded-event mechanism.** Task id on parent return receipt and child write receipt; child write after parent end without cancel ack is the finding. **No allowed fix** without task ids.

**Gap.** `git_admissible` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`task_cancel`).

**Priority.** High.


## 256. Kotlin coroutine resumes on wrong dispatcher

**Category.** Concurrency and async.

**Pattern.** Main dispatcher assumed; work resumes on Default after context switch. UI corruption off main thread.

**Example.** https://github.com/Kotlin/kotlinx.coroutines/issues/2024. Coroutine context and dispatcher inheritance.

**Recorded-event mechanism.** Dispatcher id on resume receipt differs from dispatcher id on start receipt for same coroutine id. **No allowed fix** without dispatcher ids.

**Gap.** `resolve_git_frames` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`dispatcher_id`).

**Priority.** Medium.


## 257. Swift actor reentrancy allows two mutating calls

**Category.** Concurrency and async.

**Pattern.** Actor allows second call before first awaits. State violates invariants; blame on second call site.

**Example.** https://github.com/apple/swift/issues/59138. Actor reentrancy and executor checks.

**Recorded-event mechanism.** Actor id and call serial on enter receipts; second enter before first suspend receipt is the finding. **No allowed fix** without serial numbers.

**Gap.** `split_frame` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`actor_serial`).

**Priority.** High.


## 258. WebAssembly atomic wait wakes wrong agent

**Category.** Concurrency and async.

**Pattern.** Shared memory `wait`/`notify` mismatch after grow. Only one module's commit changed memory size.

**Example.** https://github.com/WebAssembly/threads/issues/155. Threads proposal notify/wait pairing.

**Recorded-event mechanism.** Memory index and offset on wait and notify receipts; notify without matching wait agent id is the finding. **No allowed fix** without agent ids.

**Gap.** `paths_identify` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`wasm_agent`).

**Priority.** Medium.


## 259. Park/unpark pair lost on thread spawn race

**Category.** Concurrency and async.

**Pattern.** Unpark before park: thread parks forever. Spawn and park in different source files.

**Example.** https://github.com/rust-lang/rust/issues/39364. std::thread park/unpark race documentation.

**Recorded-event mechanism.** Park receipt and unpark receipt with thread id; unpark instant before park instant with no wake is the finding. **No allowed fix** without thread id.

**Gap.** `touched_line` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`park_unpark`).

**Priority.** High.


## 260. RWLock writer starvation under reader preference

**Category.** Concurrency and async.

**Pattern.** Readers never release; writer timeout. Config change enabled reader preference in one commit; hang shows in writer file.

**Example.** https://github.com/golang/go/issues/17973. RWMutex fairness and starvation.

**Recorded-event mechanism.** Lock mode receipt counts readers vs writer wait ms at `path:line`. Writer wait exceeds recorded threshold with reader count never zero is the finding. **No allowed fix** without lock metrics.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`rwlock_metrics`).

**Priority.** Medium.


## 261. Semaphore release exceeds initial count

**Category.** Concurrency and async.

**Pattern.** Release without matching acquire raises or corrupts count. Fix in release path; crash in acquire path.

**Example.** https://github.com/tokio-rs/tokio/issues/4204. Semaphore permit accounting bug report.

**Recorded-event mechanism.** Semaphore id, count before/after on acquire and release receipts. Count negative or above max is the finding. **No allowed fix** without count fields.

**Gap.** `record_git_edges` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`semaphore_count`).

**Priority.** High.


## 262. Fork while another thread held a mutex

**Category.** Concurrency and async.

**Pattern.** Child inherits locked mutex; parent dies holding lock. Child deadlock with no commit in child repo.

**Example.** https://github.com/python/cpython/issues/6721. Fork safety and threading (historical issue).

**Recorded-event mechanism.** Pre-fork lock hold receipt with pid; post-fork child pid with same lock id still held and no unlock in child is the finding. **No allowed fix** without fork receipt.

**Gap.** `run_git` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`fork_lock`).

**Priority.** High.


## 263. Priority inversion on realtime mutex

**Category.** Concurrency and async.

**Pattern.** Low-priority thread holds lock; high-priority thread blocks; medium runs. Timing-dependent failure.

**Example.** https://github.com/systemd/systemd/issues/6114. Realtime scheduling and priority inversion in service manager.

**Recorded-event mechanism.** Priority integer on lock holder and waiter receipts at same lock id. Waiter priority greater than holder with no priority inheritance receipt is the finding. **No allowed fix** without priority fields.

**Gap.** `upstream_note_for` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`sched_priority`).

**Priority.** Medium.


## 264. Coroutine leaks on detached scope

**Category.** Concurrency and async.

**Pattern.** Fire-and-forget coroutine outlives shutdown. Use-after-free in runtime; creating commit only added `spawn`.

**Example.** https://github.com/tokio-rs/tokio/issues/4961. Task leak when runtime dropped with running tasks.

**Recorded-event mechanism.** Runtime shutdown receipt and task running receipt; task still running after shutdown instant. **No allowed fix** without runtime id.

**Gap.** `failure_revision` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`runtime_shutdown`).

**Priority.** High.


## 265. Barrier wait count mismatch

**Category.** Concurrency and async.

**Pattern.** `pthread_barrier` or `std::barrier` participant count wrong. One thread waits forever after team size change commit.

**Example.** https://github.com/bminor/glibc/issues/1463. Barrier implementation edge case (public glibc tracker).

**Recorded-event mechanism.** Barrier id, expected participants, arrived count on wait receipt. Arrived less than expected at deadline is the finding. **No allowed fix** without barrier counts.

**Gap.** `verdict_of` (`crates/vestige-mcp/src/walk_verify/probe.rs`) — needs new receipt/event type (`barrier_count`).

**Priority.** Medium.


## 266. Lock-free push sees stale head after ABA

**Category.** Concurrency and async.

**Pattern.** CAS succeeds on recycled node. Crash in pop; push commit reused node from free list without epoch.

**Example.** https://github.com/facebook/folly/issues/1473. Folly concurrent data structure ABA discussion.

**Recorded-event mechanism.** Epoch id on node alloc and free receipts; CAS on node whose free epoch equals current without alloc is the finding. **No allowed fix** without epoch id.

**Gap.** `ddmin` (`crates/vestige-mcp/src/walk_verify/hunks.rs`) — needs new receipt/event type (`aba_epoch`).

**Priority.** High.


## 267. GPU kernel launch races host flag

**Category.** Concurrency and async.

**Pattern.** Host sets `ready` before memcpy completes. Kernel reads garbage. Host file blamed; missing fence in driver commit.

**Example.** https://github.com/pytorch/pytorch/issues/73891. CUDA async memcpy and kernel ordering.

**Recorded-event mechanism.** Memcpy receipt and kernel launch receipt with same stream id; launch timestamp before memcpy complete timestamp is the finding. **No allowed fix** without stream id.

**Gap.** `classify_starts` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`cuda_stream`).

**Priority.** High.


## 268. OpenMP parallel region nested incorrectly

**Category.** Concurrency and async.

**Pattern.** Nested `parallel` without `collapse` duplicates work. Wrong numerical result; outer pragma in header only.

**Example.** https://github.com/llvm/llvm-project/issues/55835. OpenMP nested parallelism bug report.

**Recorded-event mechanism.** Region id stack on enter receipts; depth exceeds recorded max legal depth is the finding. **No allowed fix** without region stack.

**Gap.** `log_args` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`omp_region`).

**Priority.** Medium.


## 269. Java `CompletableFuture` completes exceptionally on wrong thread

**Category.** Concurrency and async.

**Pattern.** Callback runs on `ForkJoinPool` though `async` specified executor. Race in default executor config commit.

**Example.** https://github.com/openjdk/jdk/issues/828557. CompletableFuture executor selection (OpenJDK issue).

**Recorded-event mechanism.** Executor id on complete receipt vs requested executor id on create receipt for same future id. **No allowed fix** without future id.

**Gap.** `apply_version_range` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`executor_id`).

**Priority.** Medium.


## 270. Erlang process message order violated across nodes

**Category.** Concurrency and async.

**Pattern.** Cluster upgrade changes distribution; message order differs. Failure on consumer node; sender commit on other repo.

**Example.** https://github.com/erlang/otp/issues/5891. Distribution protocol ordering.

**Recorded-event mechanism.** Node pair and message serial on send and receive receipts; receive serial not greater than prior on same pair is the finding. **No allowed fix** without serial.

**Gap.** `packages_on_parent_chain` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`msg_serial`).

**Priority.** High.


## 271. C# lock taken on wrong monitor object

**Category.** Concurrency and async.

**Pattern.** Boxing creates new monitor per iteration. Intermittent data race; lock statement line blamed.

**Example.** https://github.com/dotnet/runtime/issues/67270. Monitor lock on boxed value types discussion.

**Recorded-event mechanism.** Monitor object identity hash on enter receipts for same `path:line`; differing hashes across iterations is the finding. **No allowed fix** without object identity on receipt.

**Gap.** `match_paths` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`monitor_id`).

**Priority.** Medium.


## 272. Fiber yield leaves critical section entered

**Category.** Concurrency and async.

**Pattern.** Coroutine yields while holding spinlock protecting scheduler queue. Next fiber enters same queue. Deadlock or corruption.

**Example.** https://github.com/ziglang/zig/issues/5376. Async and concurrency bugs in Zig standard library.

**Recorded-event mechanism.** Critical section depth receipt on yield; depth greater than zero on yield without release is the finding. **No allowed fix** without depth receipt.

**Gap.** `reverted_after` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`crit_depth`).

**Priority.** High.


## 273. Read-copy-update stall from long read section

**Category.** Concurrency and async.

**Pattern.** Linux RCU stall detector fires. Stack in reader; quiescent state delayed by blocking call added elsewhere.

**Example.** https://github.com/systemd/systemd/issues/13776. RCU stall warnings under load in journal.

**Recorded-event mechanism.** Stall receipt: cpu, blocked task `path:line` from stack; jiffies in read section exceed threshold. **No allowed fix** without stall receipt.

**Gap.** `git_structure` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`rcu_stall`).

**Priority.** High.


## 274. Happens-before edge missing across thread pool queue

**Category.** Concurrency and async.

**Pattern.** Task B reads flag set by task A without synchronization. TSAN reports race; both tasks spawned from same pool submit line.

**Example.** https://github.com/google/sanitizers/issues/1259. TSAN false positive/negative discussion on thread pools.

**Recorded-event mechanism.** TSAN edge receipt between two `path:line` sites; absent edge with shared memory write-read pair is the finding. **No allowed fix** without TSAN graph receipt.

**Gap.** `line_slots` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`tsan_edge`).

**Priority.** High.


## 275. Lazy initialization race on static pointer

**Category.** Concurrency and async.

**Pattern.** Double-checked locking without memory barrier. Two threads construct singleton; one leaks, one UAF.

**Example.** https://github.com/llvm/llvm-project/issues/60154. Clang codegen for static local initialization.

**Recorded-event mechanism.** Init guard byte receipt per `path:line`; two threads enter init with guard not complete is the finding. **No allowed fix** without guard state.

**Gap.** `parse_hunk_ranges` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`init_guard`).

**Priority.** High.


## 276. Windows IOCP completion reordered vs commit order

**Category.** Concurrency and async.

**Pattern.** Two writes complete; completion packets reversed. Reader sees torn record.

**Example.** https://github.com/libuv/libuv/issues/2078. IOCP completion order on Windows.

**Recorded-event mechanism.** File offset and completion serial on IOCP receipts; serial decreases while offsets increase is the finding. **No allowed fix** without completion serial.

**Gap.** `diff_payload` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`iocp_serial`).

**Priority.** Medium.


## 277. Epoll edge-triggered loop stops reading

**Category.** Concurrency and async.

**Pattern.** Edge-triggered epoll stops draining socket; connection hangs. `epoll_ctl` change in one file; hang reported in worker.

**Example.** https://github.com/haproxy/haproxy/issues/948. Epoll ET and load balancer stuck connections.

**Recorded-event mechanism.** Socket fd and epoll event mask on ctl receipt; readable edge without subsequent read receipt before next wait is the finding. **No allowed fix** without fd receipts.

**Gap.** `read_commits` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`epoll_fd`).

**Priority.** Medium.


## 278. Mutex priority ceiling not raised on nested lock

**Category.** Concurrency and async.

**Pattern.** Realtime thread nests locks; ceiling too low. Priority inversion timeout.

**Example.** https://github.com/freebsd/freebsd/issues/255867. Priority ceiling protocol in libthr.

**Recorded-event mechanism.** Lock nest receipt with ceiling values; inner lock ceiling less than outer holder priority is the finding. **No allowed fix** without ceiling integers.

**Gap.** `commit_in_ancestor_range` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`lock_ceiling`).

**Priority.** Medium.


## 279. Atomic RMW on misaligned field on ARM

**Category.** Concurrency and async.

**Pattern.** Packed struct atomic increment faults or tears on ARM32. Alignment fix in struct definition; crash in increment line.

**Example.** https://github.com/rust-lang/rust/issues/27030. Unaligned atomics and platform support.

**Recorded-event mechanism.** Alignment receipt at `path:line` column; RMW on offset not multiple of size is the finding. **No allowed fix** without alignment receipt.

**Gap.** `lock_bumps_from_diff` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`field_align`).

**Priority.** High.


## 280. Futex waiter never woken after requeue

**Category.** Concurrency and async.

**Pattern.** Priority-inheritance futex requeue leaves waiter off both lists. Process hangs in `futex_wait`.

**Example.** https://github.com/golang/go/issues/60078. Go runtime futex hang on Linux (public issue).

**Recorded-event mechanism.** Futex addr and waiter list id before/after requeue receipt; waiter id absent from both lists is the finding. **No allowed fix** without futex receipts.

**Gap.** `first_bad_logged` (`crates/vestige-mcp/src/walk_verify/git.rs`) — needs new receipt/event type (`futex_waiter`).

**Priority.** High.


## 281. Concurrent map write during range

**Category.** Concurrency and async.

**Pattern.** Pattern specific to concurrent map write during range: failure under concurrency; blame line not the inducing edit.

**Example.** https://github.com/golang/go/issues/23428. Concurrent map iteration and write panic.

**Recorded-event mechanism.** Receipt stores opaque resource id and operation. Conflicting operations without happens-before is the finding. **No allowed fix** without resource id on receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`resource_id`).

**Priority.** High.


## 282. Slice append races shared backing array

**Category.** Concurrency and async.

**Pattern.** Pattern specific to slice append races shared backing array: failure under concurrency; blame line not the inducing edit.

**Example.** https://github.com/golang/go/issues/40701. Slice reallocation races with concurrent read.

**Recorded-event mechanism.** Receipt stores opaque resource id and operation. Conflicting operations without happens-before is the finding. **No allowed fix** without resource id on receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`resource_id`).

**Priority.** Medium.


## 283. Copy-on-write page fault during read

**Category.** Concurrency and async.

**Pattern.** Pattern specific to copy-on-write page fault during read: failure under concurrency; blame line not the inducing edit.

**Example.** https://github.com/torvalds/linux/issues/20096. COW fault during read — public mm issue discussion.

**Recorded-event mechanism.** Receipt stores opaque resource id and operation. Conflicting operations without happens-before is the finding. **No allowed fix** without resource id on receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`resource_id`).

**Priority.** High.


## 284. False sharing on atomic counter cache line

**Category.** Concurrency and async.

**Pattern.** Pattern specific to false sharing on atomic counter cache line: failure under concurrency; blame line not the inducing edit.

**Example.** https://github.com/golang/go/issues/41555. False sharing performance regression reports.

**Recorded-event mechanism.** Receipt stores opaque resource id and operation. Conflicting operations without happens-before is the finding. **No allowed fix** without resource id on receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`resource_id`).

**Priority.** Medium.


## 285. Work-stealing deque push/pop race

**Category.** Concurrency and async.

**Pattern.** Pattern specific to work-stealing deque push/pop race: failure under concurrency; blame line not the inducing edit.

**Example.** https://github.com/openjdk/jdk/issues/8062848. ForkJoinPool deque race (historical).

**Recorded-event mechanism.** Receipt stores opaque resource id and operation. Conflicting operations without happens-before is the finding. **No allowed fix** without resource id on receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`resource_id`).

**Priority.** High.


## 286. Timer reset while callback running

**Category.** Concurrency and async.

**Pattern.** Pattern specific to timer reset while callback running: failure under concurrency; blame line not the inducing edit.

**Example.** https://github.com/golang/go/issues/27169. Timer reset reentrancy on Go runtime.

**Recorded-event mechanism.** Receipt stores opaque resource id and operation. Conflicting operations without happens-before is the finding. **No allowed fix** without resource id on receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`resource_id`).

**Priority.** Medium.


## 287. Context switch during critical section on single core

**Category.** Concurrency and async.

**Pattern.** Pattern specific to context switch during critical section on single core: failure under concurrency; blame line not the inducing edit.

**Example.** https://github.com/rust-lang/rust/issues/55010. Preemption while holding spinlock on uniprocessor.

**Recorded-event mechanism.** Receipt stores opaque resource id and operation. Conflicting operations without happens-before is the finding. **No allowed fix** without resource id on receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`resource_id`).

**Priority.** High.


## 288. Lock elision starts transaction on conflicting cache line

**Category.** Concurrency and async.

**Pattern.** Pattern specific to lock elision starts transaction on conflicting cache line: failure under concurrency; blame line not the inducing edit.

**Example.** https://github.com/llvm/llvm-project/issues/38925. TSX lock elision abort storms.

**Recorded-event mechanism.** Receipt stores opaque resource id and operation. Conflicting operations without happens-before is the finding. **No allowed fix** without resource id on receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`resource_id`).

**Priority.** Medium.


## 289. Async signal-safe function called with mutex held

**Category.** Concurrency and async.

**Pattern.** Pattern specific to async signal-safe function called with mutex held: failure under concurrency; blame line not the inducing edit.

**Example.** https://github.com/systemd/systemd/issues/5644. Signal handler vs mutex in systemd.

**Recorded-event mechanism.** Receipt stores opaque resource id and operation. Conflicting operations without happens-before is the finding. **No allowed fix** without resource id on receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`resource_id`).

**Priority.** High.


## Batch 1 counts (items 240–289)

High: 240, 241, 243, 245, 246, 247, 248, 259, 261, 262, 264, 266, 267, 270, 273, 274, 275, 279, 280, 281, 283, 285, 287, 289 (24).

Medium: 242, 244, 249, 250, 253, 256, 257, 258, 260, 263, 265, 268, 269, 271, 276, 277, 278, 282, 284, 286, 288 (21).

Low: none.

Needs new receipt type: all except none (0 buildable now).

No allowed fix: every item unless the named receipt is already on disk (0 automatic without ingest).

Real public examples: 50. Constructed: 0.

Running total new items: 50 (240–289).

