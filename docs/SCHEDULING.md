# Scheduling Model

How `asb replay` and `asb agent` dispatch work.

## The unit of work: schedule-task

A **schedule-task** is the smallest unit the scheduler hands out:

```
schedule-task = (task, execution_index)
```

- `task` is one recorded scenario task (replay) or one prompt (agent).
- `execution_index` is the 0-based repetition number.
- Inside one schedule-task in replay mode there are `r_1 … r_n` underlying
  HTTP requests (the recorded turns). Those are replayed serially by the
  worker that pulls the schedule-task.

With `T` distinct tasks and `R = --repetitions`, a run produces
`T × R` schedule-tasks.

## The queue: ordered pending list

Before dispatch, all `T × R` schedule-tasks are materialized into a
single ordered list `L`. `--policy` picks the shape:

### `sequential` (all reps of one task, then all reps of the next)

```
L = [ (t1,0), (t1,1), …, (t1,R-1),
      (t2,0), (t2,1), …, (t2,R-1),
      …
      (tT,0), …, (tT,R-1) ]
```

Good for measuring per-task performance in isolation. Bad for realism:
the server's prefix cache gets warmed by task 1 repeatedly, so every
repetition after the first is effectively a cache hit for the shared
prefix.

### `round_robin` (one rep of each task, then next round)

```
L = [ (t1,0), (t2,0), …, (tT,0),
      (t1,1), (t2,1), …, (tT,1),
      …
      (t1,R-1), …, (tT,R-1) ]
```

Interleaves tasks so the cache churns between different task prefixes.
Reasonable default for balanced measurement.

### `random` (shuffle of sequential)

```
L = shuffle([ (t1,0), …, (tT,R-1) ])
```

Closest to "N independent users working on unrelated things at the same
time" - the hardest cache-sharing pattern for the server. Pass `--seed N`
to reproduce a specific shuffle across runs; omit it for fresh entropy.

**Agent mode defaults to `random`** because anything else lets the
agent's own prompt prefix (tool blocks, system prompt) get a free cache
ride across tasks that run back-to-back.

## The dispatcher: pool of J workers

With the list `L` built, dispatch is a literal work queue:

```
pending = deque(L)                  # head = next schedule-task
for each slot s in 0..J-1:
    spawn async worker(s):
        while pending:
            item = pending.popleft()
            run(item)                # this is the only awaited call
        return
wait for all J workers to finish
```

Key properties:

- **No batch lockstep.** A fast slot does not wait for a slow slot to
  finish before pulling the next item. Every slot is independent.
- **Exactly J long-lived workers.** Not `T × R` suspended coroutines;
  not a fresh worker per item. The httpx client (replay) or the
  subprocess fleet (agent) is sized to `J`, matching what a real
  production workload looks like.
- **FIFO head-pulling.** When a slot frees, it pulls `pending[0]`.
  Order in `L` is exactly the order of dispatch.
- **Atomic pop.** `collections.deque.popleft()` between two awaits is
  race-free in single-threaded asyncio, so two slots can never pop the
  same item.

Implemented in `agentic_swarm_bench.scenarios.schedule.run_work_queue`.

## Worked example

```
T = 6 tasks, R = 8 reps, J = 4 slots, policy = random, seed = 42
|L| = 48 schedule-tasks
```

At runtime:

```
t=0.0s | slot0:(t3,2)  slot1:(t1,0)  slot2:(t5,7)  slot3:(t2,4)
t=1.2s | slot0:(t4,1)  slot1:(t1,0)  slot2:(t5,7)  slot3:(t2,4)
                ↑ slot0 finished first, pulled next head
t=1.8s | slot0:(t4,1)  slot1:(t1,0)  slot2:(t6,3)  slot3:(t2,4)
                                            ↑ slot2 finished, pulled next
…
```

Any slot that frees up immediately pulls the next item from `L[0]`.
The run ends when `pending` is empty and all workers have returned.

## CLI flags

| Flag                | Symbol | Applies to    | Default               |
| ------------------- | ------ | ------------- | --------------------- |
| `--repetitions, -r` | R      | replay, agent | 1                     |
| `--max-concurrent`  | J      | replay, agent | 10 / 1                |
| `--policy`          |        | replay, agent | round_robin / random  |
| `--seed`            |        | replay, agent | None (system entropy) |

Replay defaults to `round_robin` and `J=10` because replays tend to be
bulk throughput sweeps. Agent defaults to `random` and `J=1` because a
single-agent serial run is the common debugging shape and `random`
prevents silent prefix-cache free-rides.

## Frequently asked

**Why not just `asyncio.gather` with a semaphore?**
It's semantically equivalent for correctness but (a) allocates `T × R`
suspended coroutines and clients, (b) depends on `asyncio.Semaphore`
being FIFO (implementation detail), and (c) doesn't read like the
one-line work-queue pseudocode. The explicit pool-of-J form is clearer
and uses one long-lived client per slot, which is more cache-realistic.

**Why does each replay slot own its own httpx client?**
Connection reuse. A real user holds a single keepalive connection for
the session; reusing one client across many schedule-tasks on the same
slot matches that. Using a fresh client per schedule-task, as the old
code did, injects a TLS handshake into every measurement.

**Is poison content affected by ordering?**
No. Poisoning is keyed by `(task.id, execution_index)`, not by
dispatch order. Shuffling changes who-runs-when; the bytes on the wire
for `(tX, r_i)` are identical regardless of where that schedule-task
ends up in `L`.
