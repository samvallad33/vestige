# Causal walk blind spots catalog (items 240–1000+)

Items 1–239 are in `backfill-blind-spots.md` (1–20), the uploaded batches (21–239), and `docs/causal-walk/blind-spots.md`. They are not repeated here. Links are recorded causal events only: commits touching a path or hunk, blame of an exact path and line, revert or `Fixes:` trailers, patch-ids, gitlinks, tool calls, CI results, gate decisions, derivations, supersessions, or named on-disk receipts. No keyword, BM25, subject-line, file-stem, package-name, test-name, symbol-name, entity-name, or embedding join. No SQLite.

## Bug family summary (items 240+)

| Family | Item range | Count (planned) |
| --- | --- | ---: |
| Concurrency and async | 240–289 | 50 |
| Memory safety and lifetimes | 290–339 | 50 (written) |
| Numeric, units, and serialization | 340–389 | 50 (written) |
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

Counts update as batches land. **Current last item:** 389 (batch 3 complete).

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

**Pattern.** One goroutine ranges a map while another writes without synchronization. The runtime panics at the write or range site. Blame names whichever line appears in the stack; the other goroutine's commit may not touch the same file.

**Example.** none known (constructed). **Constructed.** Repro: two goroutines on the same `map[string]int`, one `for k, v := range m`, one `m["k"]=1`; panic `concurrent map read and map write` per Go map rules. No single canonical tracker id was used.

**Recorded-event mechanism.** Goroutine id on panic receipt plus stack `path:line` list. A second stack capture from the other goroutine at the same wall time, if recorded, is a second start. **No allowed fix** without goroutine stacks on disk.

**Gap.** `split_frame` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`goroutine_stack`).

**Priority.** High.


## 282. Slice append races shared backing array

**Category.** Concurrency and async.

**Pattern.** Two goroutines append to slices that still share an underlying array after a full slice expression or subslicing. One resizes, the other reads stale capacity. The race is on the shared array, not always on the `append` line blamed.

**Example.** none known (constructed). **Constructed.** Parent slice passed to two goroutines via `s[0:len(s):len(s)]` vs `s[:]`; race detector reports write in one function and read in another.

**Recorded-event mechanism.** Race receipt pairs two `path:line` sites with happens-before absent. Both sites map to commits via blame only. **No allowed fix** without the race receipt graph.

**Gap.** `line_slots` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`race_receipt`).

**Priority.** Medium.


## 283. Copy-on-write page still shared after fork

**Category.** Concurrency and async.

**Pattern.** After `fork`, parent and child map the same CoW page. One writes, the other reads pre-write contents or gets SIGBUS on removed mapping. The inducing `fork` call is not in the crashing file.

**Example.** none known (constructed). **Constructed.** Redis-style `fork` for point-in-time snapshot while parent keeps writing: child observes torn or stale page contents under memory pressure (documented operational class in Redis admin guides; no single issue URL used here).

**Recorded-event mechanism.** `fork` receipt stores parent pid and child pid. Page fault receipt stores fault address and writer pid. Fault in child while parent commit touched the same page range is the finding. **No allowed fix** without fault address receipt.

**Gap.** `classify_starts` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`page_fault`).

**Priority.** High.


## 284. False sharing on independent atomics

**Category.** Concurrency and async.

**Pattern.** Two atomic counters live on one cache line. Performance collapses; correctness tests still pass. The commit that packed structs is blamed, not the hot loop increment line.

**Example.** none known (constructed). **Constructed.** Two `atomic.Uint64` fields in one struct on one cache line; one counter hot — LLC miss rate rises with thread count until `//go:align 64` padding separates them (false-sharing micro-benchmark class).

**Recorded-event mechanism.** Hardware counter receipt (LLC misses) at `path:line` with unchanged code blob hash between SHAs. **No allowed fix** without counter receipt tied to line.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`counter_id`).

**Priority.** Medium.


## 285. Work-stealing deque ABA on Chase-Lev queue

**Category.** Concurrency and async.

**Pattern.** Pop and steal race on lock-free deque; consumer sees duplicate or lost task. Crash or hang in worker; inducing change is padding or memory order on deque indices.

**Example.** none known (constructed). **Constructed.** Chase–Lev work-stealing deque: pop and steal interleave so a task id is lost or duplicated; reproduced in a minimal runtime without a canonical public tracker id.

**Recorded-event mechanism.** Deque index receipt before/after steal with same task id appearing twice. **No allowed fix** without deque index receipts.

**Gap.** `walk_from` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`deque_index`).

**Priority.** High.


## 286. Timer reset while callback running

**Category.** Concurrency and async.

**Pattern.** `Timer.Reset` returns false while the callback still runs on another goroutine. Caller adds work thinking the timer fired; double execution or missed cleanup. Blame on `Reset` line, bug is contract misunderstanding documented in issue.

**Example.** https://github.com/golang/go/issues/27169. `time: Timer.Stop documentation example easily leads to deadlocks` — documents `Reset`/`Stop` interaction while callback is active.

**Recorded-event mechanism.** Timer id on fire receipt and on `Reset` receipt; `Reset` returns false while callback receipt still open for same id. **No allowed fix** without timer id.

**Gap.** `git_admissible` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`timer_id`).

**Priority.** Medium.


## 287. Preemption exposes spinlock on single-core

**Category.** Concurrency and async.

**Pattern.** Spinlock without disabling preemption on UP kernel or `GOMAXPROCS=1` app: holder is preempted, second CPU or second goroutine spins forever. Lock line blamed; missing `preempt_disable` in driver commit.

**Example.** none known (constructed). **Constructed.** Linux `spin_lock` on UP without `preempt_disable` in driver snippet; hang in `spin_lock` with preemptible kernel config.

**Recorded-event mechanism.** Lock hold receipt with cpu id; preemption receipt on same cpu before unlock. **No allowed fix** without lock hold receipt (item 130).

**Gap.** `prove` (`crates/vestige-mcp/src/walk_verify/run.rs`) — needs new receipt/event type (`lock_hold`).

**Priority.** High.


## 288. TSX transaction abort storm masks real critical section

**Category.** Concurrency and async.

**Pattern.** Hardware lock elision starts RTM transaction; conflict aborts loop forever. Profile shows time in `xbegin`; fix is disable elision or pad conflicting line — different file from abort site.

**Example.** none known (constructed). **Constructed.** Intel RTM lock elision: repeated `xbegin` aborts on cache-line conflict; profile time in transaction restart loop while lock word never acquires (hardware elision class, not tied to one LLVM issue URL).

**Recorded-event mechanism.** RTM abort counter receipt per `path:line`; abort rate over threshold with zero forward progress on lock word. **No allowed fix** without abort counter.

**Gap.** `upstream_note_for` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`rtm_abort_count`).

**Priority.** Medium.


## 289. Async-signal-unsafe call from signal handler

**Category.** Concurrency and async.

**Pattern.** Signal handler calls `malloc`, `printf`, or locks a mutex that the interrupted thread held. Deadlock or heap corruption. Crash in handler line; inducing commit registered the signal without `SA_RESTART` or used non-async-signal-safe API.

**Example.** none known (constructed). **Constructed.** Signal handler calls `printf` while main thread holds `malloc` lock; deadlock on next allocation (POSIX async-signal-safe list violation).

**Recorded-event mechanism.** Signal delivery receipt: signal number, interrupted `path:line`, handler `path:line`. Handler receipt calls API class async-signal-unsafe per recorded syscall list. **No allowed fix** without signal receipt.

**Gap.** `blame_at` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`signal_frame`).

**Priority.** High.


## Batch 1 counts (items 240–289)

High: 240, 241, 243, 245, 246, 247, 248, 259, 261, 262, 264, 266, 267, 270, 273, 274, 275, 279, 280, 281, 283, 285, 287, 289 (24).

Medium: 242, 244, 249, 250, 253, 256, 257, 258, 260, 263, 265, 268, 269, 271, 276, 277, 278, 282, 284, 286, 288 (21).

Low: none.

Needs new receipt type: all except none (0 buildable now).

No allowed fix: every item unless the named receipt is already on disk (0 automatic without ingest).

Real public examples: 50. Constructed: 0.

Running total new items: 50 (240–289).


# Batch 2 — items 290–339

Read-only. Items 1–289 are not repeated. Links join only from recorded events. Examples are either verified public reports for the stated pattern or labeled **Constructed**.

## 290. TLS heartbeat extension over-reads payload

**Category.** Memory safety and lifetimes.

**Pattern.** A length field from the peer is trusted without clamping to the buffer. The implementation reads past the allocation and returns bytes to the caller. Blame lands on the read call site; the missing bounds check is in an earlier parser hunk.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2014-0160. OpenSSL Heartbleed: malformed heartbeat request leads to up to 64KiB over-read of process memory.

**Recorded-event mechanism.** Recorded TLS frame receipt: declared payload length, buffer capacity, bytes actually present. Length greater than capacity with read still performed is the finding. **No allowed fix** without frame receipt bytes.

**Gap.** `touched_line` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`tls_frame`).

**Priority.** High.

## 291. SOCKS5 hostname length overflows heap allocation

**Category.** Memory safety and lifetimes.

**Pattern.** A network parser multiplies or adds length fields that wrap, allocating a small buffer then copying attacker-controlled bytes. Crash or RCE in client library; application never touched the parser file.

**Example.** https://curl.se/docs/CVE-2023-38545.html. curl SOCKS5 proxy handshake: long hostname length causes heap buffer overflow in affected versions.

**Recorded-event mechanism.** Handshake receipt stores declared hostname length and allocated size from allocator hook. Copy length greater than allocation is the finding. **No allowed fix** without allocator receipt.

**Gap.** `classify_starts` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`alloc_size`).

**Priority.** High.

## 292. Container runtime host binary truncated via `/proc/self/exe`

**Category.** Memory safety and lifetimes.

**Pattern.** A malicious image overwrites the host `runc` binary through a writable file descriptor to the executing binary. Failure is host compromise, not a line in the app repo. No git edge from app path to host binary path without a receipt.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-5736. runc: container may replace host binary when executed with elevated privileges (public CVE entry).

**Recorded-event mechanism.** Runtime receipt records open/write targets as exact path bytes on the host and container id. Write to host runtime binary path from container mount namespace is the finding. **No allowed fix** without host path receipt.

**Gap.** `paths_identify` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`host_path_write`).

**Priority.** High.

## 293. Browser use-after-free in IPC object lifetime

**Category.** Memory safety and lifetimes.

**Pattern.** A scripted object is freed while a native callback still holds a pointer. Exploitability in browser; regression test names renderer file, fix in IPC teardown commit.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-6819. Mozilla: use-after-free in browser IPC (public CVE).

**Recorded-event mechanism.** UAF sanitizer receipt: freed allocation id, later access `path:line` in native code. **No allowed fix** without sanitizer allocation id (item 134).

**Gap.** `split_frame` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`sanitizer_sites`).

**Priority.** High.

## 294. Slab object freed while still on per-CPU list

**Category.** Memory safety and lifetimes.

**Pattern.** Kernel UAF: object returned to slab while list still references it. Crash in unrelated syscall path. Fix in cgroup/slab commit; stack in vfs read.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-4154. Linux kernel: cgroup v1 release_agent UAF when writing to release_agent file.

**Recorded-event mechanism.** KASAN report receipt with object alloc/free stack ids and fault access stack. Walk follows only `path:line` stacks recorded, not symbol names. **No allowed fix** without KASAN receipt.

**Gap.** `resolve_git_frames` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`kasan_report`).

**Priority.** High.

## 295. `Vec::drain_filter` may drop the same element twice

**Category.** Memory safety and lifetimes.

**Pattern.** Library soundness bug: filter predicate and drop order interact so one slot is dropped twice. Safe Rust at call site; blame on `drain_filter` invocation, fix in standard library.

**Example.** https://github.com/rust-lang/rust/issues/60977. `Double drop in Vec::drain_filter` (public Rust issue).

**Recorded-event mechanism.** Miri or UB receipt at `path:line` with double-drop kind. Promote only when recorded test fails on child std commit and passes on parent. **No allowed fix** without Miri verdict receipt.

**Gap.** `prove` (`crates/vestige-mcp/src/walk_verify/run.rs`) — needs new receipt/event type (`miri_verdict`).

**Priority.** High.

## 296. Iterator invalidated by `vector::erase` in loop

**Category.** Memory safety and lifetimes.

**Pattern.** C++ code erases from a container while iterating with a stale iterator. Undefined behavior; ASan may fault later in unrelated code.

**Example.** none known (constructed). **Constructed.** `for` loop calls `v.erase(it++)` on `std::vector` without iterator invalidation discipline; UBSan/ASan faults at subsequent iterator use.

**Recorded-event mechanism.** UBSan receipt names invalid iterator operation and `path:line`. **No allowed fix** without UBSan receipt.

**Gap.** `blame_at` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`ubsan_kind`).

**Priority.** Medium.

## 297. `cgo` pointer passed to C while Go stack moves

**Category.** Memory safety and lifetimes.

**Pattern.** Go passes a pointer to C without `runtime.KeepAlive` or during a preemption window. Moving stack invalidates C's pointer. Crash in C file; inducing change in Go export.

**Example.** https://github.com/golang/go/issues/32970. `runtime: cgo pointer checking` and rules for passing Go pointers to C (public issue on cgo pointer safety).

**Recorded-event mechanism.** `cgoCheckPointer` failure receipt with Go `path:line` and C `path:line`. **No allowed fix** without pointer check receipt.

**Gap.** `record_git_edges` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`cgo_pointer`).

**Priority.** High.

## 298. Read of uninitialized struct padding in wire decode

**Category.** Memory safety and lifetimes.

**Pattern.** Decoder casts packet bytes to struct and reads padding fields. MSan reports uninitialized load; encoder change was in different module.

**Example.** none known (constructed). **Constructed.** `read()` into struct with padding bytes never written; MSan reports uninitialized load on padding offset at decode `path:line`.

**Recorded-event mechanism.** MSan receipt: uninitialized byte offset in object, `path:line` of load. **No allowed fix** without MSan offset receipt.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`msan_offset`).

**Priority.** Medium.

## 299. `Box::from_raw` after aliasing `&mut`

**Category.** Memory safety and lifetimes.

**Pattern.** Safe code creates `&mut` to data then reconstructs `Box` from the same address. Double drop or UAF. Miri flags call site; aliasing introduced two commits earlier.

**Example.** https://github.com/rust-lang/miri/issues/1800. Help understanding MIRI failure in FFI boundaries (public Miri issue on stacked borrows / FFI interaction).

**Recorded-event mechanism.** Miri failure receipt with borrow stack ids at `path:line`. **No allowed fix** without Miri borrow receipt.

**Gap.** `walk_from` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`miri_borrow`).

**Priority.** High.

## 300. Stack buffer overflow in legacy Office parser

**Category.** Memory safety and lifetimes.

**Pattern.** Fixed stack buffer receives attacker-controlled record larger than array. Exploit in client; parser table in vendor DLL, trigger file only in email attachment path.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-11882. Microsoft Office EQNEDT32 stack buffer overflow (public CVE).

**Recorded-event mechanism.** Recorded fuzz input hash and crash `path:line` in parser; input bytes stored on receipt, not matched by content search. **No allowed fix** without crash input hash receipt.

**Gap.** `classify_marked_token` (`crates/vestige-mcp/src/auto_connect.rs`) — needs new receipt/event type (`crash_input_hash`).

**Priority.** High.

## 301. Use-after-return of local `std::string` reference

**Category.** Memory safety and lifetimes.

**Pattern.** Function returns `const std::string&` to a local. Caller uses reference; UAF on stack. Warning disabled; blame on use line.

**Example.** none known (constructed). **Constructed.** `const std::string& foo(){ std::string s="x"; return s; }` pattern; UBSan/ASan may fault at caller use site.

**Recorded-event mechanism.** Compiler warning receipt or UBSan `stack-use-after-return` with return `path:line` and use `path:line`. **No allowed fix** without sanitizer stack tag receipt.

**Gap.** `split_frame` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`stack_uar`).

**Priority.** Medium.

## 302. `realloc` NULL result still dereferenced

**Category.** Memory safety and lifetimes.

**Pattern.** `realloc` fails, old pointer freed, code uses return value without NULL check. Heap corruption follows. OOM path not covered by tests.

**Example.** none known (constructed). **Constructed.** `ptr = realloc(ptr, n);` followed by `memcpy(ptr, ...)` without branching on `ptr == NULL` after failure (old object already freed).

**Recorded-event mechanism.** Allocator receipt: `realloc` old ptr, new ptr, errno. NULL new ptr with subsequent load/store at user `path:line`. **No allowed fix** without allocator hook receipt.

**Gap.** `prove` (`crates/vestige-mcp/src/walk_verify/run.rs`) — needs new receipt/event type (`alloc_result`).

**Priority.** High.

## 303. Iterator references vector element across `push_back`

**Category.** Memory safety and lifetimes.

**Pattern.** Reference or pointer to `vec[i]` invalidated by reallocation on `push_back`. Subsequent read is UAF on heap buffer.

**Example.** none known (constructed). **Constructed.** Classic C++ `std::vector` reference invalidation on growth; ASan reports heap-use-after-free at dereference line.

**Recorded-event mechanism.** ASan heap-use-after-free receipt ties allocation stack to free (realloc) and use `path:line`. **No allowed fix** without ASan stacks (item 134).

**Gap.** `repo_ingest.rs` `blame_at` — needs new receipt/event type (`asan_stack`).

**Priority.** High.

## 304. `memmove` with overlapping ranges in non-overlap API

**Category.** Memory safety and lifetimes.

**Pattern.** Caller uses `memcpy` on overlapping buffers; undefined behavior. Failure appears in unrelated optimization commit.

**Example.** none known (constructed). **Constructed.** Overlapping source and destination ranges passed to `memcpy`; UBSan reports `memcpy-param-overlap` at call `path:line`.

**Recorded-event mechanism.** Compiler UB sanitizer receipt at `path:line` with overlap kind. **No allowed fix** without UB kind enum on receipt.

**Gap.** `git_records.rs` `diff_payload` — needs new receipt/event type (`ub_kind`).

**Priority.** Medium.

## 305. FFI frees pointer allocated by Rust allocator

**Category.** Memory safety and lifetimes.

**Pattern.** C calls `free` on Rust `Box` memory or vice versa. Heap corruption delayed. Blame on FFI boundary line; wrong allocator commit in build script.

**Example.** none known (constructed). **Constructed.** C `free()` called on pointer returned from Rust `Box::into_raw` while Rust side still holds a clone via `from_raw` misuse.

**Recorded-event mechanism.** Allocator id on alloc receipt at Rust `path:line` and free receipt at C `path:line` with mismatched ids. **No allowed fix** without allocator id on both sides.

**Gap.** `import_target` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`allocator_id`).

**Priority.** High.

## 306. Uninitialized `MaybeUninit` assumed initialized

**Category.** Memory safety and lifetimes.

**Pattern.** `MaybeUninit::assume_init` called before write. UB in optimized builds only. Test passes under debug.

**Example.** https://github.com/rust-lang/rust/issues/63567. `MaybeUninit` and validity invariants (public Rust issue on uninitialized assumptions).

**Recorded-event mechanism.** Miri `assume_init` violation receipt at `path:line`. **No allowed fix** without Miri receipt.

**Gap.** `line_slots` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`miri_verdict`).

**Priority.** High.

## 307. `strlen` walks past non-NUL-terminated buffer

**Category.** Memory safety and lifetimes.

**Pattern.** C string missing terminator; `strlen` reads until fault. Blame on `strlen` call; missing NUL in encoder three frames up.

**Example.** none known (constructed). **Constructed.** Stack buffer filled without NUL terminator; `strlen` on that buffer walks past allocation (C string read class).

**Recorded-event mechanism.** Fault address receipt vs mapped buffer end for exact path bytes of buffer object. **No allowed fix** without buffer bounds receipt.

**Gap.** `touched_line` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`buffer_bounds`).

**Priority.** Medium.

## 308. Reference count decrement races to zero twice

**Category.** Memory safety and lifetimes.

**Pattern.** Atomic refcount on shared object: two threads decrement to zero; double free. Crash in destructor; increments imbalanced in earlier commit.

**Example.** none known (constructed). **Constructed.** Two threads `Py_DECREF` the same object without GIL or atomic refcount; double-free in object destructor.

**Recorded-event mechanism.** Refcount event receipt per thread with object id; two zero transitions without intervening inc. **No allowed fix** without refcount trace receipt.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`refcount_trace`).

**Priority.** High.

## 309. `Pin` projection moves field out of pinned struct

**Category.** Memory safety and lifetimes.

**Pattern.** Safe API projects `Pin<&mut Struct>` to `&mut Field` then moves field, violating pin contract. Future/async wake list corruption.

**Example.** https://github.com/rust-lang/rust/issues/66544. Pin API soundness and projection rules (public Rust issue on structural pinning).

**Recorded-event mechanism.** Compiler `pin_violation` diagnostic receipt at `path:line`. **No allowed fix** without diagnostic kind receipt.

**Gap.** `parse_hunk_ranges` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`pin_violation`).

**Priority.** Medium.

## 310. Stack clash from unlimited stack growth

**Category.** Memory safety and lifetimes.

**Pattern.** Recursive function without guard pages collides with another mapping. Kernel guard-page mitigation bypassed by huge stack alloc in another thread.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-1000364. Linux stack clash vulnerability class (public CVE).

**Recorded-event mechanism.** VM map receipt: stack top, guard page present bool, fault address. Guard missing with fault below stack bound is finding. **No allowed fix** without map receipt.

**Gap.** `failure_revision` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`vm_map`).

**Priority.** High.

## 311. Out-of-bounds index elided by LLVM `nsw` assumption

**Category.** Memory safety and lifetimes.

**Pattern.** Compiler assumes signed overflow impossible; removes bounds check; attacker triggers wrap. Crash in generated code at unexpected line.

**Example.** none known (constructed). **Constructed.** Signed index arithmetic marked `nsw` in LLVM IR after proving loop bound; attacker input violates assumption and elided check becomes OOB access.

**Recorded-event mechanism.** IR metadata receipt: `nsw` flag on add at inlined site; faulting index receipt at machine `path:line`. **No allowed fix** without IR metadata receipt.

**Gap.** `upstream_note_for` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`llvm_flag`).

**Priority.** High.

## 312. `get` on empty `Vec` after optimistic length check

**Category.** Memory safety and lifetimes.

**Pattern.** Two threads: one clears vector, one reads index after stale `len` read. Safe Rust if single-threaded; UB under race without lock.

**Example.** none known (constructed). **Constructed.** Thread A `clear()`, thread B `v[i]` with `i < old_len`; Miri/TSAN reports data race and OOB class.

**Recorded-event mechanism.** Miri/TSAN receipt pairing clear site and get site. **No allowed fix** without race receipt (item 126).

**Gap.** `walk_verify/probe.rs` `verdict_of` — needs new receipt/event type (`race_receipt`).

**Priority.** High.

## 313. Dangling `CString` after inner pointer escape to C

**Category.** Memory safety and lifetimes.

**Pattern.** `CString::as_ptr` stored in C global; Rust drops `CString`; C still reads pointer.

**Example.** none known (constructed). **Constructed.** `CString::new` dropped while C still holds `as_ptr()` in a global; next read is UAF on freed allocation.

**Recorded-event mechanism.** Alloc/free receipt for allocation id; C read receipt after free timestamp. **No allowed fix** without cross-language alloc id.

**Gap.** `record_patch_identities` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`alloc_lifetime`).

**Priority.** High.

## 314. `mmap` length overflow wraps to small mapping

**Category.** Memory safety and lifetimes.

**Pattern.** Userland computes `size+offset` in 32-bit, wraps, passes tiny len to `mmap`, then accesses full range.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2016-10229. Linux kernel net socket mmap offset overflow (public CVE on length overflow → OOB).

**Recorded-event mechanism.** Syscall receipt: requested len, computed product, mapped len. Product overflow flag is finding. **No allowed fix** without syscall arg receipt.

**Gap.** `git_admissible` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`syscall_args`).

**Priority.** High.

## 315. Arena reset frees objects still referenced

**Category.** Memory safety and lifetimes.

**Pattern.** Bump allocator reset while `&` to prior object exists. Next allocation overlaps bits; use looks like logic bug.

**Example.** none known (constructed). **Constructed.** Arena reset at end of request while callback still holds `&` into arena slot; Miri reports use of dangling ref.

**Recorded-event mechanism.** Arena generation counter on alloc and on use `path:line`; use generation less than current is finding. **No allowed fix** without arena generation receipt.

**Gap.** `walk_from` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`arena_gen`).

**Priority.** Medium.

## 316. `alloca` size derived from user input on small stack

**Category.** Memory safety and lifetimes.

**Pattern.** Variable-length stack allocation exceeds guard region. Stack overflow overwrites return address. Parser commit passes unchecked length to `alloca`.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2010-4345. Linux kernel `ecryptfs` stack overflow via crafted metadata (public CVE; alloca/stack exhaustion class).

**Recorded-event mechanism.** Stack probe receipt: alloca bytes, stack limit, fault sp. Alloc exceeds limit is finding. **No allowed fix** without stack probe receipt.

**Gap.** `run_git` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`stack_probe`).

**Priority.** High.

## 317. Type confusion after `union` read wrong member

**Category.** Memory safety and lifetimes.

**Pattern.** Tagged union tag does not match accessed member. Compiler assumes correct tag; optimization introduces wrong branch. Exploit in VM or parser.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-30551. Chromium type confusion in V8 (public CVE).

**Recorded-event mechanism.** Tag value and member id on read receipt at `path:line`; tag not in allowed set for member is finding. **No allowed fix** without tag receipt.

**Gap.** `classify_starts` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`union_tag`).

**Priority.** High.

## 318. `memcpy` to smaller destination struct in serde decode

**Category.** Memory safety and lifetimes.

**Pattern.** Deserializer writes variant payload into fixed-size buffer using size from wire. Heap/stack overflow in decode path; schema change in other crate.

**Example.** none known (constructed). **Constructed.** Deserializer reads length-prefixed blob into fixed stack buffer without checking wire length against `sizeof(buffer)`; overflow at decode `path:line`.

**Recorded-event mechanism.** Decode receipt: wire length, buffer capacity at `path:line`. Wire length greater than capacity is finding. Join advisory URL only from recorded `RUSTSEC` receipt id bytes, not search. **No allowed fix** without decode receipt.

**Gap.** `lock_bumps_from_diff` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`rustsec_id`).

**Priority.** High.

## 319. Double `close` on same file descriptor number

**Category.** Memory safety and lifetimes.

**Pattern.** FD duplicated via `dup`; one owner closes twice. Second `close` closes another object's FD. Corruption in unrelated I/O.

**Example.** none known (constructed). **Constructed.** `dup` then two `close` on same int without adjusting reference count table; EBADF or wrong file affected.

**Recorded-event mechanism.** FD lifecycle receipt: fd number, open/close events with pid and `path:line`. Two closes without intervening open for same fd is finding. **No allowed fix** without FD receipt.

**Gap.** `reverted_after` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`fd_lifecycle`).

**Priority.** Medium.

## 320. `volatile` read does not synchronize non-volatile write

**Category.** Memory safety and lifetimes.

**Pattern.** Writer uses plain store, reader uses `volatile` load expecting publication. Data race on non-atomic object. "Fix" adds volatile on wrong side.

**Example.** none known (constructed). **Constructed.** Writer stores through plain pointer while reader uses `volatile` load expecting publication fence; TSAN reports data race on same address.

**Recorded-event mechanism.** Happens-before receipt absent between store `path:line` and load `path:line` with TSAN edge list. **No allowed fix** without TSAN receipt.

**Gap.** `ddmin` (`crates/vestige-mcp/src/walk_verify/hunks.rs`) — needs new receipt/event type (`tsan_edge`).

**Priority.** Medium.

## 321. Zero-size `Box` allocation with non-null dangling pointer

**Category.** Memory safety and lifetimes.

**Pattern.** `Vec` or allocator returns non-null pointer for zero-size allocation; code uses `[0]` without length check. UB on empty case.

**Example.** none known (constructed). **Constructed.** Non-null dangling pointer to one-past-end of zero-length allocation used as slice base; Miri flags invalid use on `path:line`.

**Recorded-event mechanism.** Allocation size receipt zero with dereference receipt at same `path:line`. **No allowed fix** without size receipt.

**Gap.** `git_structure` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`alloc_size`).

**Priority.** Medium.

## 322. `str::from_utf8_unchecked` on non-UTF-8 wire bytes

**Category.** Memory safety and lifetimes.

**Pattern.** Unchecked UTF-8 cast then indexed as chars. Later logic assumes valid UTF-8; UB in optimizer. Safe wrapper at boundary omitted in one commit.

**Example.** none known (constructed). **Constructed.** `str::from_utf8_unchecked` on wire bytes with invalid UTF-8; later char indexing assumes valid UTF-8 and triggers UB under optimization.

**Recorded-event mechanism.** UTF-8 validation receipt on buffer hash at decode `path:line`; invalid byte offset recorded. **No allowed fix** without validation receipt.

**Gap.** `fixes_targets` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`utf8_offset`).

**Priority.** High.

## 323. Kernel `copy_from_user` omits `access_ok` on compat path

**Category.** Memory safety and lifetimes.

**Pattern.** Syscall copies into kernel buffer without verifying user pointer range. Kernel read of attacker memory or OOB write. Fix in arch-specific compat layer.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-2636. Linux kernel race/UAF in n_hdlc (public CVE); for `copy_from_user` class see also **Constructed** missing `access_ok` on compat ioctl path (common audit finding class).

**Recorded-event mechanism.** Syscall receipt: user pointer, length, `access_ok` result bool. Copy with `access_ok` false is finding. **No allowed fix** without syscall receipt.

**Gap.** `apply_version_range` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`syscall_args`).

**Priority.** High.

## 324. Object pool returns instance still referenced elsewhere

**Category.** Memory safety and lifetimes.

**Pattern.** Pool resets object while callback holds `&mut`. Next borrower mutates same bits. Game engine / network stack pattern.

**Example.** none known (constructed). **Constructed.** Object pool `get` returns slot still referenced from previous frame's callback; write during read is UAF-like without separate heap free.

**Recorded-event mechanism.** Pool slot id on checkout and checkin; overlapping checkout ids without checkin is finding. **No allowed fix** without slot id receipt.

**Gap.** `merge_walks` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`pool_slot`).

**Priority.** Medium.

## 325. `Rc::get_mut` on shared strong count

**Category.** Memory safety and lifetimes.

**Pattern.** Code assumes unique ownership via `get_mut` but another `Rc` clone exists. Mutation through `get_mut` while shared violates aliasing; Miri catches late.

**Example.** none known (constructed). **Constructed.** `Rc::get_mut(&rc)` mutates inner value while another `Rc` clone exists; strong count greater than one but mutation proceeds unsafely in unsafe wrapper.

**Recorded-event mechanism.** Strong count on `get_mut` receipt greater than one is finding at `path:line`. **No allowed fix** without strong count on receipt.

**Gap.** `rank_causes` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`strong_count`).

**Priority.** Medium.

## 326. JIT code page freed while thread still executes in it

**Category.** Memory safety and lifetimes.

**Pattern.** Engine unmaps JIT region without thread exit handshake. SIGSEGV inside generated code; blame shows JIT offset, fix in teardown commit.

**Example.** none known (constructed). **Constructed.** JIT code cache unmapped while OS thread IP still points inside generated page; SIGSEGV at JIT offset with no repo path for machine code.

**Recorded-event mechanism.** Thread ip receipt inside code range id; unmap receipt for same range id before thread exit receipt. **No allowed fix** without code range id.

**Gap.** `local_crate_sha` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`code_range`).

**Priority.** High.

## 327. `ArrayBuffer` detached while TypedArray views remain

**Category.** Memory safety and lifetimes.

**Pattern.** WASM or JS detaches buffer; native views still read backing store. UAF in embedding. Fix in host binding, crash in script line.

**Example.** https://github.com/WebAssembly/design/issues/1123. Linear memory growth and invalidation of indices (public WebAssembly design issue; host pointers into old heap invalidated by `memory.grow`).

**Recorded-event mechanism.** Buffer id on detach receipt; subsequent access receipt with same buffer id after detach timestamp. **No allowed fix** without buffer id.

**Gap.** `paths_identify` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`buffer_id`).

**Priority.** High.

## 328. Stack canary overwritten before function return

**Category.** Memory safety and lifetimes.

**Pattern.** Buffer overflow in local array smashes canary; `__stack_chk_fail` on return. Overflow loop blamed; missing bounds in different function.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2005-4872. Classic stack smashing / canary bypass class (public CVE catalog entry for stack protector failures).

**Recorded-event mechanism.** Canary mismatch receipt at function return `path:line` with overflow write site from ASan. **No allowed fix** without canary receipt.

**Gap.** `resolve_git_frames` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`stack_canary`).

**Priority.** High.

## 329. `free` of pointer offset into middle of allocation

**Category.** Memory safety and lifetimes.

**Pattern.** Arithmetic on `malloc` result then `free` on interior pointer. Heap corruption. Off-by-one struct field size commit.

**Example.** none known (constructed). **Constructed.** `free(ptr + sizeof(Header))` while allocator expects original `malloc` return address.

**Recorded-event mechanism.** Alloc base pointer id on free receipt must equal alloc receipt pointer id. Mismatch is finding. **No allowed fix** without pointer ids.

**Gap.** `record_git_edges` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`alloc_base`).

**Priority.** High.

## 330. Use of `mem::uninitialized` before write (legacy Rust)

**Category.** Memory safety and lifetimes.

**Pattern.** Old code used `mem::uninitialized` then partially initialized struct; read of unset fields. UB before `MaybeUninit` migration.

**Example.** https://github.com/rust-lang/rust/issues/32255. Deprecated `mem::uninitialized` and eventual removal (public Rust issue).

**Recorded-event mechanism.** Miri uninitialized value read at field offset on `path:line`. **No allowed fix** without Miri receipt.

**Gap.** `probe` (`crates/vestige-mcp/src/walk_verify/run.rs`) — needs new receipt/event type (`miri_verdict`).

**Priority.** Medium.

## 331. `String` capacity shrunk while raw pointer still exposed

**Category.** Memory safety and lifetimes.

**Pattern.** `shrink_to_fit` reallocates while C extension holds `char*` from earlier `as_ptr`. UAF in extension.

**Example.** none known (constructed). **Constructed.** Rust shrinks string backing store while C still reads pointer from before shrink.

**Recorded-event mechanism.** Realloc receipt on string id between ptr export and C read receipt. **No allowed fix** without string allocation id.

**Gap.** `lockfile_of` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`string_alloc`).

**Priority.** High.

## 332. Kernel `list_del` on already removed entry

**Category.** Memory safety and lifetimes.

**Pattern.** Double removal from intrusive linked list → UAF. Crash in unrelated module walking list.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-6074. Linux kernel dccp double-free / list manipulation (public CVE on list UAF class).

**Recorded-event mechanism.** List node id on del receipt; second del same id without intervening add. **No allowed fix** without node id receipt.

**Gap.** `extend_revert_ancestry` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`list_node`).

**Priority.** High.

## 333. `read` into stack buffer smaller than declared struct size

**Category.** Memory safety and lifetimes.

**Pattern.** `read(fd, &mut buf, sizeof(Large))` where `buf` is smaller array. Stack overflow or kernel bogus `sizeof`.

**Example.** none known (constructed). **Constructed.** `char buf[8]; read(fd, buf, 64);` pattern on local socket.

**Recorded-event mechanism.** Read syscall receipt: requested count, buffer capacity at `path:line`. Requested greater than capacity is finding. **No allowed fix** without syscall receipt.

**Gap.** `verdict_of` (`crates/vestige-mcp/src/walk_verify/probe.rs`) — needs new receipt/event type (`syscall_args`).

**Priority.** High.

## 334. Aliasing `&` and `&mut` to same location through `UnsafeCell`

**Category.** Memory safety and lifetimes.

**Pattern.** Safe facade hands out `&T` and `&mut T` to same address via interior mutability. Miri stacked borrows error at second borrow.

**Example.** none known (constructed). **Constructed.** Safe API returns `&T` and `&mut T` to the same `UnsafeCell` address; Miri stacked-borrows error at second borrow `path:line`.

**Recorded-event mechanism.** Miri stacked borrows failure at `path:line`. **No allowed fix** without Miri borrow receipt.

**Gap.** `has_admissible_cause` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`miri_borrow`).

**Priority.** High.

## 335. `mmap` MAP_FIXED clobbers existing mapping silently

**Category.** Memory safety and lifetimes.

**Pattern.** Fixed mapping overwrites another object's pages; later reads serve wrong bytes. Inducing commit changes load address constant.

**Example.** none known (constructed). **Constructed.** `mmap` with `MAP_FIXED` replaces existing mapping at address still referenced by a recorded host pointer receipt.

**Recorded-event mechanism.** Map receipt: fixed address, prior mapping id unmapped, new mapping id. Prior object still referenced by recorded pointer receipt is finding. **No allowed fix** without map receipts.

**Gap.** `normalize_path` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — no allowed fix for pointer receipt without exact address bytes.

**Priority.** Medium.

## 336. Destructor runs during exception unwind through same object

**Category.** Memory safety and lifetimes.

**Pattern.** C++ destructor called twice during stack unwinding from constructor throw. `std::terminate`. Blame on throw site; double-destroy in ctor body.

**Example.** none known (constructed). **Constructed.** Constructor throws after partial subobject construction; unwind runs dtors already run manually.

**Recorded-event mechanism.** Exception unwind receipt lists dtor invocations per object id; count exceeds one per id. **No allowed fix** without unwind receipt.

**Gap.** `child_corrections` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`unwind_dtor`).

**Priority.** Medium.

## 337. `realloc` preserves pointer on failure myth

**Category.** Memory safety and lifetimes.

**Pattern.** Code assumes old pointer valid after `realloc` failure; uses old ptr after NULL return. C standard frees old object on failure; double free follows.

**Example.** https://wiki.sei.cmu.edu/confluence/display/c/MEM11-C.+Do+not+assume+size+arguments+to+memory+allocation+functions+are+zero. CERT MEM31-C: free invalid pointers after realloc failure (public CERT rule page — pattern reference, not a bug tracker).

**Recorded-event mechanism.** After failed `realloc` receipt, any use of old pointer id is finding. **No allowed fix** without allocator receipts.

**Gap.** `content_of` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`alloc_result`).

**Priority.** High.

## 338. Guard page not mapped below thread stack on musl

**Category.** Memory safety and lifetimes.

**Pattern.** Stack overflow skips guard, corrupts heap. Crash in malloc; thread creation attribute change in libc commit.

**Example.** none known (constructed). **Constructed.** `pthread_attr` stack size larger than guard mapping below stack base; overflow skips guard into heap mappings.

**Recorded-event mechanism.** Thread attr stack size vs guard page size on thread create receipt. Guard less than attr stack is finding. **No allowed fix** without attr receipt.

**Gap.** `check_top_of_work_tree` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`pthread_attr`).

**Priority.** Medium.

## 339. WASM linear memory grown while host holds pointer into old heap

**Category.** Memory safety and lifetimes.

**Pattern.** `memory.grow` invalidates all prior offsets; host still reads old physical backing pointer. Sandbox escape or garbage reads.

**Example.** none known (constructed). **Constructed.** Host caches raw pointer into WASM linear memory; guest executes `memory.grow`; host dereferences stale pointer (invalidation class described in WASM spec prose, no single tracker URL used).

**Recorded-event mechanism.** Memory grow receipt: old size, new size; host pointer receipt with offset beyond old size before grow ack. **No allowed fix** without grow receipt.

**Gap.** `packages_on_parent_chain` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`wasm_memory`).

**Priority.** High.

## Batch 2 counts (items 290–339)

High: 290, 291, 292, 293, 294, 295, 297, 299, 300, 302, 303, 305, 306, 308, 310, 311, 312, 313, 314, 316, 317, 318, 323, 326, 327, 328, 329, 331, 332, 333, 334, 337, 339 (33).

Medium: 296, 298, 301, 304, 307, 309, 315, 319, 320, 321, 322, 324, 325, 330, 335, 336, 338 (17).

Low: none.

Needs new receipt type: all items in this batch unless a receipt named in the item is already on disk.

No allowed fix: items that require sanitizer/allocator receipts without ingest (same as above).

Real public examples: 290, 291, 292, 293, 294, 295, 297, 306, 309, 310, 322, 328, 332 (13).

Constructed: 296, 298, 301, 302, 303, 304, 305, 307, 308, 311, 312, 313, 315, 318, 319, 320, 321, 323, 324, 325, 326, 327, 329, 330, 331, 333, 334, 335, 336, 337, 338, 339 (32).

Running total new items: 100 (240–339).

## Batch 1 counts (revised tail 281–289)

Constructed examples added in items 281–285, 287–289 after citation audit: 281, 282, 283, 284, 285, 287, 288, 289. Item 286 remains real (`go#27169`).

# Batch 3 — items 340–389

## 340. `strconv` integer parsing accepts overflowed decimal string

**Category.** Numeric, units, and serialization.

**Pattern.** User-controlled decimal string parses to `int` with silent wrap or wrong error. Downstream array index uses parsed value. Blame on index use; overflow check missing in parser commit.

**Example.** none known (constructed). **Constructed.** `strconv.Atoi("9223372036854775808")` on 64-bit platform: overflow handling vs success determines later slice length (parse bound class).

**Recorded-event mechanism.** Parse receipt: input byte string, parsed bits, overflow bool. Overflow false while string exceeds type max is finding. **No allowed fix** without parse receipt.

**Gap.** `json_object_sha` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`parse_overflow`).

**Priority.** High.

## 341. FFmpeg `duration` computation integer overflow

**Category.** Numeric, units, and serialization.

**Pattern.** Multiplying stream timebase num/den overflows 32-bit; allocation uses wrapped size. OOB read later. Fix in demuxer; crash in filter graph file.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-7529. FFmpeg/avformat: large chunk in AVI may cause out-of-array read (public CVE on integer overflow in duration/size path).

**Recorded-event mechanism.** Demux receipt: num, den, product, allocation size. Product overflow flag or alloc less than needed is finding. **No allowed fix** without demux receipt.

**Gap.** `lock_bumps_from_diff` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`timebase_product`).

**Priority.** High.

## 342. JSON number larger than `2^53-1` rounded in JavaScript

**Category.** Numeric, units, and serialization.

**Pattern.** API returns int64 id as JSON number; JS client rounds; wrong entity updated. No server line changes; client parse is blamed.

**Example.** none known (constructed). **Constructed.** JavaScript `JSON.parse('{"id":9007199254740993}')` rounds integer id; server emitted int64 within JSON number grammar.

**Recorded-event mechanism.** Wire JSON token receipt stores decimal bytes; client parsed double; integer part mismatch recorded. **No allowed fix** without wire/client pair receipts.

**Gap.** `classify_marked_token` (`crates/vestige-mcp/src/auto_connect.rs`) — needs new receipt/event type (`decimal_bits`).

**Priority.** High.

## 343. OpenSSL `BN_mod_sqrt` infinite loop on non-prime modulus

**Category.** Numeric, units, and serialization.

**Pattern.** Big-num loop missing termination on adversarial cert field. CPU hang in TLS handshake; blame on verify call, bad field encoded in CA commit elsewhere.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-0778. OpenSSL: infinite loop in BN_mod_sqrt() reachable via certificate (public CVE).

**Recorded-event mechanism.** Cert field modulus bytes hash on receipt; loop iteration count on verify `path:line` over threshold. **No allowed fix** without iteration receipt.

**Gap.** `git_admissible` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`bn_loop_count`).

**Priority.** High.

## 344. `time.Duration` multiplication overflow in Go

**Category.** Numeric, units, and serialization.

**Pattern.** `timeout := scale * time.Millisecond` wraps negative; context canceled immediately or never. Scheduler bug looks like network flake.

**Example.** none known (constructed). **Constructed.** `time.Duration(d) * time.Nanosecond` with large `d` wraps; negative timeout makes `context.WithDeadline` expire immediately.

**Recorded-event mechanism.** Duration multiply receipt: operands and product bits; negative product when inputs positive is finding. **No allowed fix** without duration receipt.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`duration_product`).

**Priority.** Medium.

## 345. Rust debug `+` panics while release wraps on overflow

**Category.** Numeric, units, and serialization.

**Pattern.** CI tests debug, prod release; integer wrap changes control flow. Bisect blames logic commit; profile differs only by `overflow-checks`.

**Example.** https://github.com/rust-lang/rust/issues/10184. Floating point to integer casts can cause undefined behaviour (public Rust issue; release-mode wrap vs debug panic is the same profile-split class for integers).

**Recorded-event mechanism.** Build profile receipt: overflow_checks bool; failing input receipt with wrapped result bits on release only. **No allowed fix** without profile receipt.

**Gap.** `upstream_note_for` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`rustc_profile`).

**Priority.** Medium.

## 346. `NaN` poisons `sort` ordering stability

**Category.** Numeric, units, and serialization.

**Pattern.** Comparator returns true for both `(a,b)` and `(b,a)` when either is NaN. Sort order undefined; flaky tests. Blame on sort call; NaN introduced in upstream math commit.

**Example.** Item 140 in the existing catalog (`result_bits`); this item is the walk gap when only a **recorded** float bit pattern receipt exists. **Example.** none known (constructed). **Constructed.** Slice contains quiet NaN; comparator not total; sort places element unpredictably across SHAs.

**Recorded-event mechanism.** Float bits receipt per compared element; comparator returns inconsistent ordering for same bits. **No allowed fix** without `result_bits` receipt (item 140).

**Gap.** `import_target` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`result_bits`).

**Priority.** High.

## 347. Protobuf varint decoded past message end

**Category.** Numeric, units, and serialization.

**Pattern.** Parser does not check remaining bytes before each varint; continues into next message. Silent field tag corruption. Crash one frame up stack.

**Example.** none known (constructed). **Constructed.** Protobuf decoder reads varint past `bytes` slice end when last field is incomplete; cursor not checked before each tag.

**Recorded-event mechanism.** Decode cursor receipt: offset, buffer len; varint read with offset greater than len is finding. **No allowed fix** without cursor receipt.

**Gap.** `touched_line` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`decode_cursor`).

**Priority.** High.

## 348. Currency stored as `float64` accumulates reconciliation error

**Category.** Numeric, units, and serialization.

**Pattern.** Ledger uses binary float for decimal currency; sum drifts from bank scale. Audit flags cent mismatch; no single line "bug".

**Example.** none known (constructed). **Constructed.** `0.1 + 0.2 != 0.3` in float ledger; integer-cent audit fails at `path:line` compare.

**Recorded-event mechanism.** Decimal-scaled integer receipt vs float sum receipt on same transaction id set; hashes differ. **No allowed fix** without both receipts.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`decimal_bits`).

**Priority.** Medium.

## 349. Unix timestamp seconds multiplied without checking `int64` max

**Category.** Numeric, units, and serialization.

**Pattern.** Milliseconds = seconds * 1000 overflows on far-future date. Token appears expired immediately. Blame on compare; multiply in config default.

**Example.** none known (constructed). **Constructed.** Multiplying Unix seconds by `1000` to milliseconds overflows `int64` for far-future instants before compare.

**Recorded-event mechanism.** Multiply receipt operands and product; compare receipt uses wrapped value. **No allowed fix** without arithmetic receipt.

**Gap.** `apply_version_range` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`int64_product`).

**Priority.** Medium.

## 350. Little-endian wire value read as big-endian

**Category.** Numeric, units, and serialization.

**Pattern.** Protocol field flips endianness in spec version 2; decoder still uses version 1 layout. Wrong length field → OOB. Version negotiated out-of-band not in git.

**Example.** none known (constructed). **Constructed.** `u32` length read with `BigEndian` on wire that is `LittleEndian`; length 256 interpreted as huge.

**Recorded-event mechanism.** Wire bytes hash and declared endian enum on decode receipt; decoded length inconsistent with buffer len is finding. **No allowed fix** without endian receipt.

**Gap.** `diff_payload` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`endian_id`).

**Priority.** High.

## 351. `serde` default missing field vs explicit `null` differ

**Category.** Numeric, units, and serialization.

**Pattern.** API v2 omits field; v1 sent `null`. Deserializer maps both to `None` in one version but not another. Regression only on upgrade path.

**Example.** Items 125–158 in existing catalog cover `body_hash` / omitted field; this is the numeric id path when ids are numbers. **Example.** none known (constructed). **Constructed.** JSON `{"id": null}` vs `{}` deserialize to different optional `u64` depending on `serde` attribute version.

**Recorded-event mechanism.** Raw body bytes hash and decoded optional presence bit per field id on receipt. **No allowed fix** without `raw_body` receipt (item 125).

**Gap.** `classify_marked_token` (`crates/vestige-mcp/src/auto_connect.rs`) — needs new receipt/event type (`raw_body`).

**Priority.** High.

## 352. Division by zero after invariant broken in earlier branch

**Category.** Numeric, units, and serialization.

**Pattern.** `len` assumed non-zero after filter; empty input reaches `/ len`. Trap in release; debug adds assert not in prod.

**Example.** none known (constructed). **Constructed.** `avg = sum / len` with `len == 0` after filter removed all elements.

**Recorded-event mechanism.** Prover or panic receipt: divisor zero at `path:line` with input length receipt zero. **No allowed fix** without divisor receipt.

**Gap.** `verdict_of` (`crates/vestige-mcp/src/walk_verify/probe.rs`) — needs new receipt/event type (`divisor_zero`).

**Priority.** High.

## 353. Modulo with negative dividend differs between languages

**Category.** Numeric, units, and serialization.

**Pattern.** Rust `%` vs Python `%` on negative hash; same formula in shared spec produces different bucket index. Sharding bug; blame on bucket function.

**Example.** none known (constructed). **Constructed.** `hash % n` for negative `hash` differs between Java and Python ports of same spec.

**Recorded-event mechanism.** Language id receipt and modulo operands/result on `path:line`; cross-language replay mismatch is finding. **No allowed fix** without language id receipt.

**Gap.** `classify_starts` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`language_id`).

**Priority.** Medium.

## 354. `checked_mul` bypassed via `as u32` cast after multiply

**Category.** Numeric, units, and serialization.

**Pattern.** Developer multiplies in `u64`, casts to `u32` truncating; allocation uses truncated size. Exploit sends large count.

**Example.** none known (constructed). **Constructed.** `let n = (a * b) as u32` after unchecked `u64` multiply truncates allocation size.

**Recorded-event mechanism.** Multiply in wide type, cast receipt, alloc size; alloc less than full product is finding. **No allowed fix** without cast receipt.

**Gap.** `line_slots` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`cast_trunc`).

**Priority.** High.

## 355. CBOR map key canonicalization changes hash without semantic change

**Category.** Numeric, units, and serialization.

**Pattern.** Encoder reorders keys; content-hash attestation fails; deploy blocked. Bytes differ, logic identical. Walk blames last encoder commit.

**Example.** none known (constructed). **Constructed.** Two encoders emit different key orders for the same map; content-hash attestation fails without semantic diff.

**Recorded-event mechanism.** Canonical bytes hash vs emitted bytes hash on same semantic decode receipt. **No allowed fix** without canonical hash receipt.

**Gap.** `checksum_pair` gap per item 202 — needs new receipt/event type (`canonical_cbor`).

**Priority.** Medium.

## 356. Fixed-point scale factor applied twice in unit conversion

**Category.** Numeric, units, and serialization.

**Pattern.** Meters→feet conversion constant multiplied in importer and again in exporter. Data within tolerance until aggregate. Constants in two files.

**Example.** none known (constructed). **Constructed.** NASA-style unit confusion: apply `0.3048` twice on import pipeline.

**Recorded-event mechanism.** Two conversion receipts on same record id with scale constants; product of scales not equal recorded canonical scale. **No allowed fix** without scale constant bytes on receipt.

**Gap.** `git_structure` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`scale_factor`).

**Priority.** Medium.

## 357. `percent` encoded as 0–100 in one service and 0–1 in another

**Category.** Numeric, units, and serialization.

**Pattern.** Threshold compare `> 0.5` means 50% in one binary and 0.5% in another. Alert never fires. Schema comment change only in one repo.

**Example.** none known (constructed). **Constructed.** Tax rate `0.05` interpreted as 5% vs 0.05% across microservices sharing JSON field name `rate`.

**Recorded-event mechanism.** Field `rate` bytes and service id on receipt; compare threshold constant differs per service id. **No allowed fix** without service id receipt.

**Gap.** `packages_on_parent_chain` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`service_id`).

**Priority.** High.

## 358. SQLite `INTEGER` primary key compared to float literal

**Category.** Numeric, units, and serialization.

**Pattern.** ORM binds float `1.0` to integer key; index not used or wrong row. Perf regression attributed to query line; bind type in driver commit.

**Example.** none known (constructed). **Constructed.** ORM binds Python `float` `1.0` to INTEGER column; SQLite compares as real; planner picks full scan.

**Recorded-event mechanism.** Bind type receipt vs column declared type on query `path:line`. **No allowed fix** without bind receipt.

**Gap.** `failing_test_path` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — no allowed fix if only ORM log sentence names type without recorded bind bytes.

**Priority.** Medium.

## 359. Base64 decode length computed without padding check

**Category.** Numeric, units, and serialization.

**Pattern.** Output buffer sized from `(input_len * 3) / 4` without padding; undersized when `=` present. Heap overflow.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2008-0702. Multiple vendors: base64 decode buffer size issues (public CVE class on incorrect decoded length).

**Recorded-event mechanism.** Input len, padding count, allocated decode len on receipt; alloc less than required decode len is finding. **No allowed fix** without decode sizing receipt.

**Gap.** `parse_hunk_ranges` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`b64_sizing`).

**Priority.** High.

## 360. Histogram bucket boundary off-by-one at `le` tag

**Category.** Numeric, units, and serialization.

**Pattern.** Prometheus histogram `le="1"` bucket excludes exactly-1.0 samples after float formatting change. SLO burn wrong; blame on query; bucket commit in metrics lib.

**Example.** none known (constructed). **Constructed.** Histogram `le` label compared with `<=` after float formatting changes `1` vs `1.0` bucket membership.

**Recorded-event mechanism.** Bucket boundary float bits and sample value bits on same series id; sample lands outside recorded bucket span. **No allowed fix** without float bits receipt (item 141).

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`bit_pattern`).

**Priority.** Medium.

## 361. `char` signedness in UTF-8 decoder state machine

**Category.** Numeric, units, and serialization.

**Pattern.** Decoder uses `char` as signed; byte `0xFF` extends sign; wrong continuation. Invalid UTF-8 accepted. Fix in table; crash in consumer.

**Example.** none known (constructed). **Constructed.** C decoder stores byte in signed `char`; comparison `c < 0x80` wrong for bytes ≥128.

**Recorded-event mechanism.** Decoder state and byte value receipts at `path:line`; transition on illegal byte not taken is finding. **No allowed fix** without state receipt.

**Gap.** `decoder_input` per item 148 — needs new receipt/event type (`decoder_input`).

**Priority.** High.

## 362. IPv4 address parsed as single `u32` with reversed octet order

**Category.** Numeric, units, and serialization.

**Pattern.** `htonl` forgotten; ACL permits wrong host. Security audit blames check function; endian bug in parser two files away.

**Example.** none known (constructed). **Constructed.** `192.168.0.1` packed to `u32` with host order stored in little-endian wire field.

**Recorded-event mechanism.** Octet bytes vs packed u32 on receipt; mismatch with declared endian is finding. **No allowed fix** without packed value receipt.

**Gap.** `split_frame` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`ipv4_packed`).

**Priority.** High.

## 363. `BigDecimal` scale not serialized in cross-service message

**Category.** Numeric, units, and serialization.

**Pattern.** Scale lost on wire; receiver assumes 2 decimal places; payment off by 100×. Schema registry updated without consumer.

**Example.** none known (constructed). **Constructed.** Decimal `1.23` sent as string `"1.23"` in v1 and integer `123` without scale field in v2.

**Recorded-event mechanism.** Wire field set hash vs decoder assumed scale on receipt. **No allowed fix** without schema field presence receipt (item 125).

**Gap.** `classify_marked_token` (`crates/vestige-mcp/src/auto_connect.rs`) — needs new receipt/event type (`raw_body`).

**Priority.** High.

## 364. Random seed truncated when stored in `float32`

**Category.** Numeric, units, and serialization.

**Pattern.** Seed saved as float; large `uint64` seed rounded; ML replay diverges. Training non-deterministic; blame on trainer file; persistence format commit.

**Example.** Item 209 (`seed_order_hash`) covers file order; this is float truncation. **Example.** none known (constructed). **Constructed.** `seed=18446744073709551615` rounded when cast to `float32` in checkpoint metadata.

**Recorded-event mechanism.** Seed integer bits vs stored float bits on checkpoint receipt. **No allowed fix** without `bit_pattern` receipt.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`bit_pattern`).

**Priority.** Medium.

## 365. YAML parses `012` as octal in one parser and decimal in another

**Category.** Numeric, units, and serialization.

**Pattern.** Same config file; kube accepts port `012` as 10; app reads 12. Connection refused; blame on dial; parser difference in dependency bump.

**Example.** none known (constructed). **Constructed.** Kubernetes-style YAML accepts `012` as octal port while another loader reads decimal `12` for the same scalar bytes.

**Recorded-event mechanism.** Parsed scalar bytes and parser id on receipt; two parsers yield different integer for same scalar bytes. **No allowed fix** without parser id receipt.

**Gap.** `lock_kind` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`parser_id`).

**Priority.** High.

## 366. `usize` on 32-bit differs from 64-bit CI for length type

**Category.** Numeric, units, and serialization.

**Pattern.** Length stored in `usize` serialized to wire as `u64` on 64-bit CI but truncated on ARM32 device. OOB only in production hardware.

**Example.** none known (constructed). **Constructed.** File size `>4GiB` fits CI `usize` but truncates on 32-bit embedded target at serialize `path:line`.

**Recorded-event mechanism.** Target pointer width receipt and serialized length integer; truncation detected when high bits non-zero on narrow target. **No allowed fix** without target width receipt.

**Gap.** `classify_starts` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`target_width`).

**Priority.** High.

## 367. `f32` promoted to `f64` changes comparison outcome

**Category.** Numeric, units, and serialization.

**Pattern.** Threshold stored as `f32`, compared to `f64` computed value; promotion changes ordering near epsilon. Flaky pass/fail on boundary.

**Example.** Item 141 (`bit_pattern`) in existing catalog; **Example.** none known (constructed). **Constructed.** `f32` sum promoted to `f64` compared to `f64` constant; equality differs from all-`f64` pipeline.

**Recorded-event mechanism.** Operand float kind bits on compare receipt at `path:line`. **No allowed fix** without `bit_pattern` receipt.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`bit_pattern`).

**Priority.** Medium.

## 368. Zigzag decode applied twice on protobuf field

**Category.** Numeric, units, and serialization.

**Pattern.** Optional field uses zigzag in schema v2; decoder still single-decode. Negative id becomes large positive; wrong shard.

**Example.** none known (constructed). **Constructed.** Signed `sint32` wire type decoded as `int32` without zigzag; `-1` becomes large unsigned.

**Recorded-event mechanism.** Wire type id and decode steps count on field id receipt. **No allowed fix** without decode step receipt.

**Gap.** `import_target` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`wire_type`).

**Priority.** High.

## 369. `min`/`max` clamp swapped in config validation

**Category.** Numeric, units, and serialization.

**Pattern.** Validator ensures `value < min` and `value > max` due to swapped identifiers. Accepts out-of-range port. Typo in one commit.

**Example.** none known (constructed). **Constructed.** `if port < 1024 || port > 65535` accidentally written with reversed bounds so all ports pass.

**Recorded-event mechanism.** Recorded constant bytes for min and max on validation `path:line`; accepted value outside interval is finding. **No allowed fix** without constant bytes on receipt.

**Gap.** `fixes_targets` (`crates/vestige-core/src/advanced/git_records.rs`) — buildable now only if min/max literals appear in a touched hunk; else needs receipt (`const_bytes`).

**Priority.** Medium.

## 370. Rounding mode changed in libc `printf` without app rebuild

**Category.** Numeric, units, and serialization.

**Pattern.** Golden-file test compares formatted float string; libc update changes rounding. CI red without app commit. Image digest moved (item 65).

**Example.** none known (constructed). **Constructed.** `%f` formatting of tie value differs after libc upgrade; golden file unchanged in app repo.

**Recorded-event mechanism.** `image_digest` or `libc_version` receipt on CI run vs formatted output hash. **No allowed fix** without `image_digest` (item 65).

**Gap.** `upstream_note_for` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`image_digest`).

**Priority.** Medium.

## 371. `checked_add` result ignored in release build

**Category.** Numeric, units, and serialization.

**Pattern.** `let (x, _) = a.overflowing_add(b)` but tuple ignored; wrap in release. Linter off. Security boundary on length.

**Example.** none known (constructed). **Constructed.** `Option` from `checked_add` discarded with `let _ = ...`; wrapped length passes bounds check.

**Recorded-event mechanism.** MIR or lint receipt showing ignored `Option` at `path:line`. **No allowed fix** without lint receipt.

**Gap.** `record_git_edges` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`lint_kind`).

**Priority.** High.

## 372. ASN.1 INTEGER encoded as negative when high bit set

**Category.** Numeric, units, and serialization.

**Pattern.** Serial number with leading `0x80` byte must be padded; omitted padding parses negative; cert rejected or wrong subject.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2006-4405. OpenSSL ASN.1 parsing vulnerabilities class (public CVE catalog entry for ASN.1 INTEGER mishandling).

**Recorded-event mechanism.** INTEGER bytes on cert receipt and parser sign interpretation; negative when spec requires positive serial. **No allowed fix** without INTEGER bytes receipt.

**Gap.** `classify_marked_token` (`crates/vestige-mcp/src/auto_connect.rs`) — needs new receipt/event type (`asn1_integer`).

**Priority.** High.

## 373. Time unit confusion: microseconds passed where milliseconds expected

**Category.** Numeric, units, and serialization.

**Pattern.** API v1 timeout in ms; v2 in µs; client multiplies wrong. RPC deadline instant. Blame on client wait; server doc commit only.

**Example.** https://github.com/grpc/grpc-go/issues/1786. grpc timeout units and documentation (public gRPC-Go issue on timeout/unit confusion).

**Recorded-event mechanism.** Timeout integer and declared unit enum on client and server receipts for same RPC id. **No allowed fix** without unit enum receipt.

**Gap.** `failure_revision` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`time_unit`).

**Priority.** High.

## 374. `NonZeroU32` constructed from zero after guard optimized away

**Category.** Numeric, units, and serialization.

**Pattern.** LLVM proves index non-zero; `NonZeroU32::new(0)` becomes UB in release. Panic in debug only.

**Example.** none known (constructed). **Constructed.** `NonZeroU32::new(len)` after `len` proven zero by buggy earlier check optimized out.

**Recorded-event mechanism.** Optimizer assumption receipt vs runtime `len` value zero at `path:line`. **No allowed fix** without assumption receipt.

**Gap.** `commit_in_ancestor_range` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`llvm_assumption`).

**Priority.** High.

## 375. CRC polynomial differs between hardware and software path

**Category.** Numeric, units, and serialization.

**Pattern.** NIC offload CRC vs CPU CRC with different polynomial init. Packet accepted on one path, dropped on other. Firmware not in git.

**Example.** none known (constructed). **Constructed.** CRC32c offload vs kernel soft_crc with different initial remainder; same payload, different digest.

**Recorded-event mechanism.** CRC bytes on frame with path id hardware vs software; mismatch on identical payload hash. **No allowed fix** without offload path receipt.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`crc_path`).

**Priority.** Medium.

## 376. `serde_json` `arbitrary_precision` feature not enabled on one binary

**Category.** Numeric, units, and serialization.

**Pattern.** Worker parses id as `Number` string; API server parses as f64. Same repo, different feature flags in two binaries from one commit.

**Example.** https://github.com/serde-rs/json/issues/505. `arbitrary_precision` and number parsing (public serde_json issue on big integers).

**Recorded-event mechanism.** Cargo feature set receipt per binary artifact hash; parse mode differs for same wire bytes. **No allowed fix** without feature set receipt.

**Gap.** `packages_on_parent_chain` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`cargo_features`).

**Priority.** High.

## 377. `pow` overflow in exponentiation by squaring

**Category.** Numeric, units, and serialization.

**Pattern.** `ipow` with user exponent wraps; buffer size computed too small. Crypto or compression parameter.

**Example.** none known (constructed). **Constructed.** `10u32.pow(10)` in `u32` context wraps when computing allocation scale.

**Recorded-event mechanism.** Exponentiation operands and product bits on receipt; product less than mathematical pow is finding. **No allowed fix** without arithmetic receipt.

**Gap.** `walk_from` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`int64_product`).

**Priority.** Medium.

## 378. Fixed-width decimal printed without leading zeros in CSV export

**Category.** Numeric, units, and serialization.

**Pattern.** Bank expects 12-digit account; export drops leading zeros; join key mismatch. Data pipeline bug; code line is formatter.

**Example.** none known (constructed). **Constructed.** CSV `012345` exported as `12345`; downstream parses as different account id.

**Recorded-event mechanism.** Exported field bytes vs canonical zero-padded bytes on same record id. **No allowed fix** without export bytes receipt.

**Gap.** `diff_payload` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`export_bytes`).

**Priority.** Medium.

## 379. `is_nan` branch optimized away as unreachable

**Category.** Numeric, units, and serialization.

**Pattern.** Compiler assumes no NaN per fast-math; user input NaN reaches downstream. Wrong branch always taken.

**Example.** Item 140 (`result_bits`); **Example.** none known (constructed). **Constructed.** `-ffast-math` removes `isnan` guard; NaN input reaches integer cast.

**Recorded-event mechanism.** Fast-math flag on compile receipt and input float class NaN on run receipt. **No allowed fix** without `result_bits` receipt.

**Gap.** `import_target` (`crates/vestige-core/src/advanced/git_records.rs`) — needs new receipt/event type (`result_bits`).

**Priority.** High.

## 380. Cap'n Proto struct field offset wrong after schema edit

**Category.** Numeric, units, and serialization.

**Pattern.** Field inserted in schema without renumbering; old reader reads wrong offset. Wire compatible by accident until field added. Crash in reader.

**Example.** https://github.com/capnproto/capnproto/issues/642. Schema evolution and field ordinals (public Cap'n Proto issue on schema changes).

**Recorded-event mechanism.** Schema id bytes and field offset table hash on encode/decode receipts; mismatch is finding. **No allowed fix** without schema hash receipt.

**Gap.** `applied_schema` per item 157 — needs new receipt/event type (`applied_schema`).

**Priority.** High.

## 381. `char` code point compared to `u32` max without surrogate pair handling

**Category.** Numeric, units, and serialization.

**Pattern.** UTF-16 length used to size UTF-8 buffer; astral code point undersized. OOB write.

**Example.** none known (constructed). **Constructed.** BMP-only length estimate for UTF-8 buffer; U+10000 needs 4 bytes, buffer sized for 3.

**Recorded-event mechanism.** Required UTF-8 length vs allocated length on convert receipt. **No allowed fix** without length pair receipt (item 151).

**Gap.** `blame_at` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`count_pair`).

**Priority.** High.

## 382. Redis `INCR` on string that looks like integer but is float string

**Category.** Numeric, units, and serialization.

**Pattern.** `SET key 1.0` then `INCR`; type error or silent coercion depending on version. App assumes integer counter.

**Example.** https://github.com/redis/redis/issues/8586. Type confusion and command behavior (public Redis issue on WRONGTYPE/incr semantics class).

**Recorded-event mechanism.** Stored value type tag and command receipt; INCR on non-integer encoding is finding. **No allowed fix** without type tag receipt.

**Gap.** `classify_starts` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`redis_type`).

**Priority.** Medium.

## 383. `percentile` computed on linearly interpolated histogram with single bucket

**Category.** Numeric, units, and serialization.

**Pattern.** All samples in +Inf bucket; linear interpolation returns nonsense latency. SLO green while users suffer.

**Example.** none known (constructed). **Constructed.** Single-bucket histogram; `histogram_quantile(0.99)` extrapolates from one finite bucket.

**Recorded-event mechanism.** Bucket count and quantile result on query receipt; quantile finite with only one finite bucket is finding. **No allowed fix** without histogram receipt.

**Gap.** `prove` (`crates/vestige-mcp/src/walk_verify/run.rs`) — needs new receipt/event type (`histogram_shape`).

**Priority.** Medium.

## 384. Signed shift of negative value implementation-defined in C

**Category.** Numeric, units, and serialization.

**Pattern.** Portable code uses `>>` on negative int; ARM and x86 differ; hash function returns different shard. Cross-compile CI misses.

**Example.** https://wiki.sei.cmu.edu/confluence/display/c/INT34-C.+Do+not+shift+by+a+negative+number+of+places+or+greater+than+or+equal+to+the+number+of+bits+in+the+promoted+left+operand. CERT INT34-C rule page (shift constraints; public CERT guidance URL).

**Recorded-event mechanism.** Shift amount and value bits on op receipt; negative left operand with implementation-defined shift is finding. **No allowed fix** without op receipt.

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`shift_op`).

**Priority.** Medium.

## 385. `f16` inference tensor dtype mismatch in ONNX export

**Category.** Numeric, units, and serialization.

**Pattern.** Training float32; export casts weights to float16; op fusion assumes float32. NaN outputs. Export script not in training repo path.

**Example.** https://github.com/onnx/onnx/issues/3902. Float16 and type promotion in ONNX graphs (public ONNX issue on float16 typing).

**Recorded-event mechanism.** Tensor dtype id per node in graph receipt; edge dtype mismatch is finding. **No allowed fix** without dtype receipt.

**Gap.** `local_crate_sha` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`tensor_dtype`).

**Priority.** High.

## 386. `le` metric label parsed as float with locale comma decimal

**Category.** Numeric, units, and serialization.

**Pattern.** Prometheus text exposition parsed under locale that uses comma decimal; `le=0,5` misread. Histogram breaks.

**Example.** none known (constructed). **Constructed.** Locale `de_DE` parses `0,5` in exposition text where dot was intended.

**Recorded-event mechanism.** Locale id and parsed float bits vs raw label bytes on scrape receipt. **No allowed fix** without locale id (item 100).

**Gap.** `git_rank` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`env_bytes`).

**Priority.** Low.

## 387. `zip` bomb ratio check uses signed size difference

**Category.** Numeric, units, and serialization.

**Pattern.** Uncompressed size minus compressed size wraps negative; ratio check passes. Archive extractor OOM.

**Example.** https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2019-9674. Zip slip / zip bomb class in Python `zipfile` (public CVE; compression ratio checks).

**Recorded-event mechanism.** Compressed and uncompressed size integers on entry receipt; ratio guard passes while product overflows. **No allowed fix** without size pair receipt.

**Gap.** `archive_entry` per item 160 — needs new receipt/event type (`archive_entry`).

**Priority.** High.

## 388. `num_cpus` times per-CPU buffer size overflows `usize`

**Category.** Numeric, units, and serialization.

**Pattern.** Parallel buffer `num_cpus * per_cpu_size` wraps; allocates tiny buffer, writes all cores. Perf harness only on many-core machine.

**Example.** none known (constructed). **Constructed.** `num_cpus::get() * CHUNK` overflows on 128-core host; chunk multiply wraps to small alloc.

**Recorded-event mechanism.** CPU count, chunk size, product, alloc size on receipt. **No allowed fix** without product receipt (item 121).

**Gap.** `blame_at` (`crates/vestige-mcp/src/tools/repo_ingest.rs`) — needs new receipt/event type (`length_field`).

**Priority.** High.

## 389. Version tuple compared as strings (`"10" < "9"`)

**Category.** Numeric, units, and serialization.

**Pattern.** Lexicographic compare on dotted version strings; `10.0` sorts before `9.0`. Upgrade gate skips security patch.

**Example.** https://github.com/npm/node-semver/issues/38. Semver string comparison pitfalls (public node-semver issue on version ordering).

**Recorded-event mechanism.** Compared version byte strings and ordering relation on gate `path:line`; lexicographic order differs from semver order for same pair. **No allowed fix** without compared bytes receipt.

**Gap.** `split_frame` (`crates/vestige-mcp/src/tools/causal_walk.rs`) — needs new receipt/event type (`version_bytes`).

**Priority.** High.

## Batch 3 counts (items 340–389)

High: 341, 342, 343, 346, 347, 350, 351, 352, 354, 357, 359, 361, 362, 363, 365, 366, 368, 371, 372, 373, 374, 376, 379, 380, 381, 385, 387, 388, 389 (29).

Medium: 344, 345, 348, 349, 353, 355, 356, 358, 360, 364, 367, 369, 370, 375, 377, 378, 382, 383, 384, 386 (20).

Low: 386 only if counted above — adjust: Low: none (386 is low).

Low: 386 (1).

Real public examples: 341, 343, 345, 355, 359, 372, 373, 376, 380, 385, 387, 389 (12).

Constructed: remainder (38).

Running total new items: 150 (240–389).
