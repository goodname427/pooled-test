"""
Pooled Testing ABTest
=====================
Compare Baseline / Adaptive Pooled / Eager Pooled (Owner-aware) on static sample arrays.

Input  : a static list of samples (owner_id, true_p, toxic, arrival_time)
Output : per-strategy metrics and a side-by-side ABTest report over multiple
         parameter combinations.
"""

import math
import random
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Dict


# ---------------------------------------------------------------------------
# Sample / Owner model
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    sid: int
    toxic: int            # 1 = toxic, 0 = clean (ground truth)
    arrival_time: float
    owner_id: int = -1
    true_p: float = 0.0   # ground-truth toxic probability (system cannot see)
    # filled by simulator
    test_start_time: float = -1.0
    finish_time: float = -1.0


@dataclass
class Owner:
    oid: int
    mu_anchor: float    # long-term anchor toxic rate (fixed)
    mu_current: float   # current-stage center, drifts via OU process
    sigma: float        # per-owner std for sample-level p (short-term jitter)
    kappa: float        # mean-reversion strength for stage transitions
    drift_sigma: float  # std of stage-transition jump
    next_switch_idx: int = 0  # sample index at which next stage switch occurs

    def sample_p(self, rng: random.Random) -> float:
        # truncated normal in [0.001, 0.99]
        p = rng.gauss(self.mu_current, self.sigma)
        return max(0.001, min(0.99, p))

    def advance_stage(self, rng: random.Random):
        """OU-style mean-reverting jump: pull toward anchor + symmetric noise."""
        delta = self.kappa * (self.mu_anchor - self.mu_current) + rng.gauss(0.0, self.drift_sigma)
        self.mu_current = max(0.001, min(0.999, self.mu_current + delta))


def make_owners(
    n_owners: int,
    pop_mu: float,
    pop_sigma: float,
    owner_sigma: float,
    seed: int = 0,
    kappa: float = 0.3,
    drift_sigma: float = 0.02,
) -> List[Owner]:
    """Create owners. Each owner has anchor mu drawn from N(pop_mu, pop_sigma).
    mu_current starts at the anchor and drifts over time via an OU process."""
    rng = random.Random(seed)
    owners: List[Owner] = []
    for i in range(n_owners):
        mu = rng.gauss(pop_mu, pop_sigma)
        mu = max(0.005, min(0.5, mu))
        owners.append(Owner(
            oid=i, mu_anchor=mu, mu_current=mu,
            sigma=owner_sigma, kappa=kappa, drift_sigma=drift_sigma,
        ))
    return owners


def generate_samples_with_owners(
    n_samples: int,
    arrival_rate: float,
    owners: List[Owner],
    seed: int = 0,
    drift_rate: float = 50.0,
) -> List[Sample]:
    """Poisson arrivals + OU drift on owner toxic rates.

    Each owner has an independent stage-switch counter following a Poisson
    process with mean ``drift_rate`` samples per stage. At each switch, the
    owner's mu_current makes a mean-reverting jump toward its anchor.
    Within a stage, per-sample p ~ N(mu_current, owner.sigma).
    """
    rng = random.Random(seed)
    drift_rng = random.Random(seed * 7919 + 13)

    # initialise next-switch sample index per owner (geometric ~ Poisson process)
    for o in owners:
        gap = max(1, int(drift_rng.expovariate(1.0 / drift_rate)))
        o.next_switch_idx = gap

    # per-owner local sample counter (drives stage switches)
    owner_local_count: Dict[int, int] = {o.oid: 0 for o in owners}

    samples: List[Sample] = []
    t = 0.0
    for i in range(n_samples):
        t += rng.expovariate(arrival_rate)
        owner = owners[rng.randrange(len(owners))]
        owner_local_count[owner.oid] += 1
        if owner_local_count[owner.oid] >= owner.next_switch_idx:
            owner.advance_stage(drift_rng)
            gap = max(1, int(drift_rng.expovariate(1.0 / drift_rate)))
            owner.next_switch_idx = owner_local_count[owner.oid] + gap
        true_p = owner.sample_p(rng)
        toxic = 1 if rng.random() < true_p else 0
        samples.append(Sample(
            sid=i, toxic=toxic, arrival_time=t,
            owner_id=owner.oid, true_p=true_p,
        ))
    return samples


# Backward-compatible: old global-p generator (kept for sanity tests)
def generate_samples(
    n_samples: int,
    arrival_rate: float,
    p_func: Callable[[float], float],
    seed: int = 0,
) -> List[Sample]:
    rng = random.Random(seed)
    samples: List[Sample] = []
    t = 0.0
    for i in range(n_samples):
        t += rng.expovariate(arrival_rate)
        p_t = max(0.0, min(1.0, p_func(t)))
        toxic = 1 if rng.random() < p_t else 0
        samples.append(Sample(sid=i, toxic=toxic, arrival_time=t,
                              owner_id=0, true_p=p_t))
    return samples


# ---------------------------------------------------------------------------
# Reagent pool (M parallel slots, each test costs T)
# ---------------------------------------------------------------------------
class ReagentPool:
    """Min-heap of next-available times for M reagent slots."""

    def __init__(self, m: int):
        self.heap = [0.0] * m
        heapq.heapify(self.heap)

    def schedule(self, ready_time: float, duration: float) -> Tuple[float, float]:
        slot_free = heapq.heappop(self.heap)
        start = max(slot_free, ready_time)
        finish = start + duration
        heapq.heappush(self.heap, finish)
        return start, finish


# ---------------------------------------------------------------------------
# Owner-aware online estimator (Beta-Binomial / Laplace smoothing)
# ---------------------------------------------------------------------------
class OwnerEstimator:
    """Per-owner toxic-rate estimator with exponential decay.

    To track non-stationary (drifting) toxic rates, every observation is
    weighted by ``decay`` (in (0, 1]); old observations fade exponentially.
    Keeps (k_w, n_w) as decayed sums; p_hat = (k_w + 1) / (n_w + 2).
    Cold start uses a global decayed prior.

    decay = 1.0  -> no decay (pure cumulative posterior)
    decay = 0.95 -> recent ~20 samples dominate
    decay = 0.90 -> recent ~10 samples dominate
    """

    def __init__(self, p_init: float = 0.1, decay: float = 0.95):
        self.p_init = p_init
        self.decay = decay
        self.global_k = 0.0
        self.global_n = 0.0
        self.per_owner: Dict[int, Tuple[float, float]] = {}

    def p_hat(self, owner_id: int) -> float:
        # owner-level posterior with global fallback
        if owner_id in self.per_owner:
            k, n = self.per_owner[owner_id]
            if n >= 3.0:
                return (k + 1.0) / (n + 2.0)
            # blend with global prior when owner has too few samples
            gk, gn = self.global_k, self.global_n
            blended_k = k + (gk + 1.0) / (gn + 2.0) * 3.0
            blended_n = n + 3.0
            return blended_k / (blended_n + 1.0)
        if self.global_n > 0:
            return (self.global_k + 1.0) / (self.global_n + 2.0)
        return self.p_init

    def update(self, owner_id: int, toxic: int):
        k, n = self.per_owner.get(owner_id, (0.0, 0.0))
        self.per_owner[owner_id] = (k * self.decay + toxic,
                                    n * self.decay + 1.0)
        self.global_k = self.global_k * self.decay + toxic
        self.global_n = self.global_n * self.decay + 1.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    name: str
    n_samples: int
    total_tests: int
    makespan: float
    avg_waiting: float
    max_waiting: float
    avg_completion: float
    max_completion: float
    tests_per_sample: float
    throughput: float


def compute_metrics(name: str, samples: List[Sample], total_tests: int) -> Metrics:
    waits = [s.test_start_time - s.arrival_time for s in samples]
    comps = [s.finish_time - s.arrival_time for s in samples]
    makespan = max(s.finish_time for s in samples) - min(s.arrival_time for s in samples)
    n = len(samples)
    return Metrics(
        name=name,
        n_samples=n,
        total_tests=total_tests,
        makespan=makespan,
        avg_waiting=sum(waits) / n,
        max_waiting=max(waits),
        avg_completion=sum(comps) / n,
        max_completion=max(comps),
        tests_per_sample=total_tests / n,
        throughput=n / makespan if makespan > 0 else float("inf"),
    )


# ---------------------------------------------------------------------------
# Strategy 1: Baseline -- one sample per test
# ---------------------------------------------------------------------------
def run_baseline(samples: List[Sample], M: int, T: float) -> Metrics:
    pool = ReagentPool(M)
    total_tests = 0
    for s in sorted(samples, key=lambda x: x.arrival_time):
        start, finish = pool.schedule(s.arrival_time, T)
        s.test_start_time = start
        s.finish_time = finish
        total_tests += 1
    return compute_metrics("Baseline", samples, total_tests)


# ---------------------------------------------------------------------------
# Strategy 2: Adaptive Pooled (legacy: global p, recursive binary split)
# ---------------------------------------------------------------------------
def optimal_group_size(p_hat: float, lo: int = 2, hi: int = 32) -> int:
    if p_hat <= 1e-6:
        return hi
    if p_hat >= 0.3:
        return 1
    n_star = int(math.ceil(1.0 / math.sqrt(p_hat)))
    return max(lo, min(hi, n_star))


def run_adaptive_pooled(
    samples: List[Sample],
    M: int,
    T: float,
    tau: float,
    p_init: float = 0.1,
    window: int = 200,
) -> Metrics:
    sorted_samples = sorted(samples, key=lambda x: x.arrival_time)
    arrival_idx = 0
    n_total = len(sorted_samples)

    history: List[int] = []

    def p_hat() -> float:
        if not history:
            return p_init
        recent = history[-window:]
        return (sum(recent) + 1.0) / (len(recent) + 2.0)

    slots = [0.0] * M
    fresh_queue: List[Sample] = []
    followup_queue: List[List[Sample]] = []
    in_flight: List[Tuple[float, int, List[Sample]]] = []
    seq_counter = [0]
    total_tests = 0
    now = 0.0

    def take_arrivals_until(t: float):
        nonlocal arrival_idx
        while arrival_idx < n_total and sorted_samples[arrival_idx].arrival_time <= t:
            fresh_queue.append(sorted_samples[arrival_idx])
            arrival_idx += 1

    def fresh_oldest_wait(t: float) -> float:
        return (t - fresh_queue[0].arrival_time) if fresh_queue else 0.0

    def try_dispatch_one() -> bool:
        nonlocal total_tests
        slot_idx = min(range(M), key=lambda i: slots[i])
        slot_free = slots[slot_idx]
        t_now = max(slot_free, now)
        take_arrivals_until(t_now)

        if followup_queue:
            group = followup_queue.pop(0)
            finish = t_now + T
            slots[slot_idx] = finish
            seq_counter[0] += 1
            heapq.heappush(in_flight, (finish, seq_counter[0], group))
            total_tests += 1
            return True

        if not fresh_queue:
            return False

        n_star = optimal_group_size(p_hat())
        ready = (
            len(fresh_queue) >= n_star
            or fresh_oldest_wait(t_now) >= tau
            or arrival_idx >= n_total
        )
        if not ready:
            return False

        grp_size = min(n_star, len(fresh_queue))
        group = [fresh_queue.pop(0) for _ in range(grp_size)]
        for s in group:
            s.test_start_time = t_now
        finish = t_now + T
        slots[slot_idx] = finish
        seq_counter[0] += 1
        heapq.heappush(in_flight, (finish, seq_counter[0], group))
        total_tests += 1
        return True

    def resolve_group(group: List[Sample], finish: float):
        if len(group) == 1:
            s = group[0]
            s.finish_time = finish
            history.append(s.toxic)
            return
        any_toxic = any(s.toxic for s in group)
        if not any_toxic:
            for s in group:
                s.finish_time = finish
                history.append(s.toxic)
            return
        mid = len(group) // 2
        followup_queue.append(group[:mid])
        followup_queue.append(group[mid:])

    while arrival_idx < n_total or fresh_queue or followup_queue or in_flight:
        while try_dispatch_one():
            pass
        next_arrival = sorted_samples[arrival_idx].arrival_time if arrival_idx < n_total else float("inf")
        next_finish = in_flight[0][0] if in_flight else float("inf")
        if next_finish == float("inf") and next_arrival == float("inf"):
            if fresh_queue or followup_queue:
                now = max(now, min(slots))
                if not try_dispatch_one():
                    break
                continue
            break
        if next_finish <= next_arrival:
            finish_t, _seq, group = heapq.heappop(in_flight)
            now = finish_t
            resolve_group(group, finish_t)
        else:
            now = next_arrival

    return compute_metrics("Adaptive", samples, total_tests)


# ---------------------------------------------------------------------------
# Strategy 3: Eager Pooled (Owner-aware)
#
# Design:
#   - eager dispatch: whenever a slot is free AND queue non-empty, dispatch now
#   - FIFO ordering by arrival_time
#   - greedy lookahead group sizing:
#       walk samples from queue head, accumulate q_k = prod(1 - p_hat_i);
#       cost_k = (2 - q_k) * T / k  (per-sample expected cost, parallel fallback)
#       pick the k that MINIMIZES cost_k (stop when adding the next sample
#       would increase cost, OR when cost_k > T which means singleton is better)
#   - one-layer fallback: positive group -> all members tested individually
#     (each sample's worst-case service time = 2T)
#   - online OwnerEstimator updated on every resolved sample
# ---------------------------------------------------------------------------
def _expected_cost_per_sample(p_hats: List[float], T: float, M: int) -> Tuple[int, float]:
    """Find k in [1, len(p_hats)] minimizing per-sample expected cost.

    M-aware cost model:
        cost_k = (T + (1 - q) * ceil(k / M) * T) / k
        where q = prod(1 - p_i)
    Fallback occupies ceil(k/M) parallel rounds when triggered.

    Returns (best_k, best_cost). Singleton baseline = (1, T).
    """
    best_k = 1
    best_cost = T
    q = 1.0
    for k, p in enumerate(p_hats, start=1):
        q *= max(0.0, 1.0 - p)
        fallback_rounds = (k + M - 1) // M
        cost = (T + (1.0 - q) * fallback_rounds * T) / k
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_k = k
    return best_k, best_cost


def _adaptive_tau(arrival_rate: float, p_hat: float, T: float,
                  tau_min: float = 0.2, tau_max: float = 5.0) -> float:
    """Adaptive tau heuristic.

    Intuition:
      - high arrival rate -> queue fills fast -> small tau is enough
      - low p_hat         -> bigger groups pay off -> can wait longer
      - high p_hat        -> singletons preferred -> small tau
      - tau scales with T (so T=1 normalisation works for any T)

    Formula:
        tau = clip( (1 - p_hat) / max(arrival_rate, eps) * T, tau_min, tau_max )
    """
    eps = 1e-3
    raw = (1.0 - p_hat) / max(arrival_rate, eps) * T
    return max(tau_min, min(tau_max, raw))


def run_eager_pooled(
    samples: List[Sample],
    M: int,
    T: float,
    p_init: float = 0.15,
    max_group: int = 16,
    cold_start_n: int = 20,
    cold_start_max_group: int = 3,
    single_test_threshold: float = 0.35,
    tau: Optional[float] = None,
    arrival_rate: Optional[float] = None,
    decay: float = 0.95,
    tau_min: float = 0.2,
    tau_max: float = 5.0,
) -> Metrics:
    """Eager + Owner-aware + M-aware lookahead group sizing + 1-layer fallback.

    Improvements over the original Eager:
      - cold-start guard: while global_n < cold_start_n, cap group size at
        cold_start_max_group to avoid catastrophic over-batching
      - M-aware cost model: fallback rounds = ceil(k/M)
      - high-toxicity escape: head sample p_hat >= single_test_threshold -> k=1
      - tau soft-wait: when current queue length < lookahead-optimal k, hold
        the slot for up to tau (per head sample) to gather more arrivals;
        triggers immediate dispatch on any of:
          * head waiting >= tau
          * queue length >= optimal k
          * no more arrivals expected (flush)
    """
    sorted_samples = sorted(samples, key=lambda x: x.arrival_time)
    arrival_idx = 0
    n_total = len(sorted_samples)

    estimator = OwnerEstimator(p_init=p_init, decay=decay)

    # if arrival_rate not provided, infer from sample stream
    if arrival_rate is None and n_total >= 2:
        span = sorted_samples[-1].arrival_time - sorted_samples[0].arrival_time
        arrival_rate = (n_total - 1) / span if span > 0 else 1.0
    elif arrival_rate is None:
        arrival_rate = 1.0

    slots = [0.0] * M
    fresh_queue: List[Sample] = []
    # fallback queue: known-positive groups -> test each member individually
    fallback_queue: List[Sample] = []
    in_flight: List[Tuple[float, int, List[Sample], bool]] = []
    seq_counter = [0]
    total_tests = 0
    now = 0.0

    def take_arrivals_until(t: float):
        nonlocal arrival_idx
        while arrival_idx < n_total and sorted_samples[arrival_idx].arrival_time <= t:
            fresh_queue.append(sorted_samples[arrival_idx])
            arrival_idx += 1

    def try_dispatch_one() -> bool:
        nonlocal total_tests
        slot_idx = min(range(M), key=lambda i: slots[i])
        slot_free = slots[slot_idx]
        t_now = max(slot_free, now)
        take_arrivals_until(t_now)

        # priority 1: fallback singletons (already-known-positive groups split out)
        if fallback_queue:
            s = fallback_queue.pop(0)
            finish = t_now + T
            slots[slot_idx] = finish
            seq_counter[0] += 1
            heapq.heappush(in_flight, (finish, seq_counter[0], [s], True))
            total_tests += 1
            return True

        if not fresh_queue:
            return False

        # eager: dispatch immediately, no waiting
        # cold-start guard: shrink upper bound until global_n is large enough
        eff_max_group = (cold_start_max_group
                         if estimator.global_n < cold_start_n
                         else max_group)
        head_p_hats = [estimator.p_hat(s.owner_id)
                       for s in fresh_queue[:eff_max_group]]
        # high-toxicity escape: head too risky -> singleton
        if head_p_hats and head_p_hats[0] >= single_test_threshold:
            best_k = 1
        else:
            best_k, _cost = _expected_cost_per_sample(head_p_hats, T, M)

        # tau soft-wait: if queue cannot reach the lookahead-optimal k yet,
        # and head waiting < tau, and more arrivals are still possible -> hold
        # tau is adaptive (when not provided): scales inversely with arrival
        # rate and head p_hat
        if tau is None:
            head_p = head_p_hats[0] if head_p_hats else p_init
            cur_tau = _adaptive_tau(arrival_rate, head_p, T, tau_min, tau_max)
        else:
            cur_tau = tau
        head_wait = t_now - fresh_queue[0].arrival_time
        if (len(fresh_queue) < best_k
                and head_wait < cur_tau
                and arrival_idx < n_total):
            return False

        grp_size = min(best_k, len(fresh_queue))
        group = [fresh_queue.pop(0) for _ in range(grp_size)]
        for s in group:
            s.test_start_time = t_now
        finish = t_now + T
        slots[slot_idx] = finish
        seq_counter[0] += 1
        heapq.heappush(in_flight, (finish, seq_counter[0], group, False))
        total_tests += 1
        return True

    def resolve_group(group: List[Sample], finish: float, is_fallback: bool):
        if is_fallback or len(group) == 1:
            s = group[0]
            s.finish_time = finish
            estimator.update(s.owner_id, s.toxic)
            return
        any_toxic = any(s.toxic for s in group)
        if not any_toxic:
            for s in group:
                s.finish_time = finish
                estimator.update(s.owner_id, s.toxic)
            return
        # one-layer fallback: enqueue every member as a singleton fallback test
        # (preserve their original test_start_time = first-touch waiting moment)
        for s in group:
            fallback_queue.append(s)

    while arrival_idx < n_total or fresh_queue or fallback_queue or in_flight:
        while try_dispatch_one():
            pass
        next_arrival = sorted_samples[arrival_idx].arrival_time if arrival_idx < n_total else float("inf")
        next_finish = in_flight[0][0] if in_flight else float("inf")
        if next_finish == float("inf") and next_arrival == float("inf"):
            if fresh_queue or fallback_queue:
                now = max(now, min(slots))
                if not try_dispatch_one():
                    break
                continue
            break
        if next_finish <= next_arrival:
            finish_t, _seq, group, is_fb = heapq.heappop(in_flight)
            now = finish_t
            resolve_group(group, finish_t, is_fb)
        else:
            now = next_arrival

    return compute_metrics("Eager", samples, total_tests)


# ---------------------------------------------------------------------------
# ABTest harness
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    n_samples: int
    arrival_rate: float
    M: int
    T: float
    tau: float
    seed: int = 0
    # Owner-model params
    n_owners: int = 10
    pop_mu: float = 0.1     # owners' mu ~ N(pop_mu, pop_sigma)
    pop_sigma: float = 0.05
    owner_sigma: float = 0.02  # within-owner per-sample p std
    # OU drift params (non-stationary toxic rates)
    drift_rate: float = 50.0   # avg samples per owner-stage
    kappa: float = 0.3         # mean-reversion strength
    drift_sigma: float = 0.02  # per-stage mu jump std


def clone_samples(samples: List[Sample]) -> List[Sample]:
    return [Sample(
        sid=s.sid, toxic=s.toxic, arrival_time=s.arrival_time,
        owner_id=s.owner_id, true_p=s.true_p,
    ) for s in samples]


def run_scenario(sc: Scenario) -> Tuple[Metrics, Metrics, Metrics]:
    owners = make_owners(sc.n_owners, sc.pop_mu, sc.pop_sigma, sc.owner_sigma,
                         seed=sc.seed * 1009 + 7,
                         kappa=sc.kappa, drift_sigma=sc.drift_sigma)
    samples = generate_samples_with_owners(sc.n_samples, sc.arrival_rate,
                                           owners, seed=sc.seed,
                                           drift_rate=sc.drift_rate)
    base = run_baseline(clone_samples(samples), sc.M, sc.T)
    adaptive = run_adaptive_pooled(clone_samples(samples), sc.M, sc.T, sc.tau)
    # Eager: tau=None -> adaptive; pass arrival_rate so the estimator can
    # compute a reasonable wait window before history is accumulated.
    eager = run_eager_pooled(
        clone_samples(samples), sc.M, sc.T,
        tau=None, arrival_rate=sc.arrival_rate,
    )
    return base, adaptive, eager


def fmt_row(label: str, vals: List[str], widths: List[int]) -> str:
    cells = [label.ljust(widths[0])]
    cells += [v.rjust(w) for v, w in zip(vals, widths[1:])]
    return " | ".join(cells)


def print_report(scenario_name: str, base: Metrics, adaptive: Metrics, eager: Metrics):
    print(f"\n=== Scenario: {scenario_name} ===")
    widths = [20, 12, 12, 12, 14, 14]
    print(fmt_row("Metric",
                  [base.name, adaptive.name, eager.name, "A vs B%", "E vs B%"],
                  widths))
    print("-" * (sum(widths) + 3 * 5))

    def row(label, b, a, e, fmt="{:.3f}"):
        d_a = (a - b) / b * 100.0 if b != 0 else 0.0
        d_e = (e - b) / b * 100.0 if b != 0 else 0.0
        sa = "+" if d_a > 0 else ""
        se = "+" if d_e > 0 else ""
        print(fmt_row(label,
                      [fmt.format(b), fmt.format(a), fmt.format(e),
                       f"{sa}{d_a:.1f}%", f"{se}{d_e:.1f}%"],
                      widths))

    row("makespan", base.makespan, adaptive.makespan, eager.makespan)
    row("avg_waiting", base.avg_waiting, adaptive.avg_waiting, eager.avg_waiting)
    row("max_waiting", base.max_waiting, adaptive.max_waiting, eager.max_waiting)
    row("avg_completion", base.avg_completion, adaptive.avg_completion, eager.avg_completion)
    row("max_completion", base.max_completion, adaptive.max_completion, eager.max_completion)
    row("total_tests", base.total_tests, adaptive.total_tests, eager.total_tests, "{:.0f}")
    row("tests/sample", base.tests_per_sample, adaptive.tests_per_sample, eager.tests_per_sample)
    row("throughput", base.throughput, adaptive.throughput, eager.throughput)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    scenarios: List[Scenario] = [
        # ---- homogeneous owners (small variance) ----
        Scenario("homo_low_p",        n_samples=500, arrival_rate=2.0, M=2, T=1.0, tau=2.0,
                 n_owners=10, pop_mu=0.05, pop_sigma=0.005, owner_sigma=0.005, seed=1),
        Scenario("homo_mid_p",        n_samples=500, arrival_rate=2.0, M=2, T=1.0, tau=2.0,
                 n_owners=10, pop_mu=0.10, pop_sigma=0.005, owner_sigma=0.005, seed=2),
        Scenario("homo_high_p",       n_samples=500, arrival_rate=2.0, M=2, T=1.0, tau=2.0,
                 n_owners=10, pop_mu=0.30, pop_sigma=0.005, owner_sigma=0.005, seed=3),

        # ---- heterogeneous owners (the main motivation for owner-aware) ----
        Scenario("hetero_wide",       n_samples=500, arrival_rate=2.0, M=2, T=1.0, tau=2.0,
                 n_owners=10, pop_mu=0.10, pop_sigma=0.08, owner_sigma=0.02, seed=4),
        Scenario("hetero_extreme",    n_samples=500, arrival_rate=2.0, M=2, T=1.0, tau=2.0,
                 n_owners=10, pop_mu=0.15, pop_sigma=0.12, owner_sigma=0.02, seed=5),

        # ---- few owners, distinct profiles ----
        Scenario("few_owners_mixed",  n_samples=500, arrival_rate=2.0, M=2, T=1.0, tau=2.0,
                 n_owners=4, pop_mu=0.15, pop_sigma=0.10, owner_sigma=0.015, seed=6),

        # ---- resource scarcity ----
        Scenario("M=1_hetero",        n_samples=500, arrival_rate=2.0, M=1, T=1.0, tau=2.0,
                 n_owners=10, pop_mu=0.10, pop_sigma=0.06, owner_sigma=0.02, seed=7),
        Scenario("M=4_hetero",        n_samples=500, arrival_rate=2.0, M=4, T=1.0, tau=2.0,
                 n_owners=10, pop_mu=0.10, pop_sigma=0.06, owner_sigma=0.02, seed=8),

        # ---- arrival regimes ----
        Scenario("burst_hetero",      n_samples=500, arrival_rate=5.0, M=2, T=1.0, tau=1.0,
                 n_owners=10, pop_mu=0.10, pop_sigma=0.06, owner_sigma=0.02, seed=9),
        Scenario("sparse_hetero",     n_samples=200, arrival_rate=0.3, M=2, T=1.0, tau=3.0,
                 n_owners=10, pop_mu=0.10, pop_sigma=0.06, owner_sigma=0.02, seed=10),

        # ---- non-stationary toxic rates (OU drift) ----
        Scenario("drift_slow",        n_samples=600, arrival_rate=2.0, M=2, T=1.0, tau=2.0,
                 n_owners=10, pop_mu=0.10, pop_sigma=0.05, owner_sigma=0.01,
                 drift_rate=80.0, kappa=0.3, drift_sigma=0.02, seed=11),
        Scenario("drift_fast",        n_samples=600, arrival_rate=2.0, M=2, T=1.0, tau=2.0,
                 n_owners=10, pop_mu=0.10, pop_sigma=0.05, owner_sigma=0.01,
                 drift_rate=20.0, kappa=0.3, drift_sigma=0.04, seed=12),
        Scenario("drift_volatile",    n_samples=600, arrival_rate=2.0, M=2, T=1.0, tau=2.0,
                 n_owners=8,  pop_mu=0.12, pop_sigma=0.06, owner_sigma=0.015,
                 drift_rate=15.0, kappa=0.2, drift_sigma=0.06, seed=13),
    ]

    summary: List[Tuple[str, Metrics, Metrics, Metrics]] = []
    for sc in scenarios:
        base, adaptive, eager = run_scenario(sc)
        print_report(sc.name, base, adaptive, eager)
        summary.append((sc.name, base, adaptive, eager))

    # final aggregated table
    print("\n\n=== ABTest Summary  (lower is better; values: Baseline -> Adaptive -> Eager) ===")
    hdr = ["scenario", "tests/sample", "avg_completion", "avg_wait", "max_wait", "winner"]
    widths = [20, 22, 22, 22, 22, 12]
    print(fmt_row(hdr[0], hdr[1:], widths))
    print("-" * (sum(widths) + 3 * 5))
    for name, b, a, e in summary:
        tps = f"{b.tests_per_sample:.2f}/{a.tests_per_sample:.2f}/{e.tests_per_sample:.2f}"
        ac = f"{b.avg_completion:.2f}/{a.avg_completion:.2f}/{e.avg_completion:.2f}"
        aw = f"{b.avg_waiting:.2f}/{a.avg_waiting:.2f}/{e.avg_waiting:.2f}"
        mw = f"{b.max_waiting:.2f}/{a.max_waiting:.2f}/{e.max_waiting:.2f}"
        # winner among the three by avg_completion (primary), break tie by total_tests
        cands = [("Baseline", b), ("Adaptive", a), ("Eager", e)]
        cands.sort(key=lambda x: (x[1].avg_completion, x[1].total_tests))
        winner = cands[0][0]
        print(fmt_row(name, [tps, ac, aw, mw, winner], widths))


if __name__ == "__main__":
    main()
