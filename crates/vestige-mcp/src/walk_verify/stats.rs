//! The statistics of `--flaky`: exact tests, no normal approximations.
//!
//! * [`fisher_p`]: one-sided Fisher exact test, the upper tail of the
//!   hypergeometric distribution. It decides whether the two ends differ at
//!   all and, at the end, whether the first bad commit does.
//! * [`Wald`]: Wald's sequential probability ratio test between the two
//!   failure rates measured on the ends. It decides one commit with as few
//!   runs as the evidence allows.
//! * [`rate_lower`] and [`rate_upper`]: exact (Clopper-Pearson) bounds on a
//!   failure rate, found by bisection on the binomial CDF. They give the
//!   "at least N times more likely" figure.
//!
//! Each function is the reference tool's function of the same name with its
//! arithmetic in the same order: the binomial CDF sums its terms with the
//! compensated summation Python's `sum()` uses, the solve is 60 halvings of
//! `[0, 1]`. Binomial coefficients are exact integers while they fit 128
//! bits (any count of up to 131 runs in all) and floats from there on,
//! where the reference tool keeps exact integers; the difference is below
//! 1e-12 relative for the run counts the command accepts, against decision
//! thresholds of 0.001 and 0.01.

/// `--baseline-max` and `--strength-runs` may not exceed this. Up to here
/// every binomial coefficient the tests need is a finite float.
pub(super) const MAX_FIXED_RUNS: u64 = 500;

/// The two ends differ, or the first bad commit matters, below this p.
pub(super) const BASELINE_P: f64 = 0.001;

/// The REPEATED rung holds below this p.
pub(super) const REPEATED_P: f64 = 0.01;

/// Confidence of each one-sided rate bound: 97.5%, so the pair is a 95%
/// statement.
const CONFIDENCE: f64 = 0.975;

/// Python 3.12's `sum()` of floats: Neumaier's compensated summation.
#[derive(Default)]
struct Sum {
    total: f64,
    carry: f64,
}

impl Sum {
    fn add(&mut self, x: f64) {
        let next = self.total + x;
        self.carry += if self.total.abs() >= x.abs() {
            (self.total - next) + x
        } else {
            (x - next) + self.total
        };
        self.total = next;
    }

    fn value(&self) -> f64 {
        if self.carry != 0.0 && self.carry.is_finite() {
            self.total + self.carry
        } else {
            self.total
        }
    }
}

/// The binomial coefficient C(n, k) as a float: exact (the integer, rounded
/// once) while it fits 128 bits, a running product of floats beyond that.
pub(super) fn choose(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut exact: u128 = 1;
    let mut float: Option<f64> = None;
    for j in 1..=k {
        let factor = n - k + j;
        float = match float {
            None => {
                // C(n-k+j, j) = C(n-k+j-1, j-1) * factor / j, an integer.
                // With the common part of factor and j taken out of both,
                // what is left of j divides the running value, so dividing
                // first is exact and overflows only when the result does.
                let common = gcd(factor, j);
                let reduced = exact / u128::from(j / common);
                match reduced.checked_mul(u128::from(factor / common)) {
                    Some(next) => {
                        exact = next;
                        None
                    }
                    None => Some(exact as f64 * factor as f64 / j as f64),
                }
            }
            Some(value) => Some(value * factor as f64 / j as f64),
        };
    }
    float.unwrap_or(exact as f64)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// P(X <= k) for X ~ Binomial(n, p).
pub(super) fn binom_cdf(k: u64, n: u64, p: f64) -> f64 {
    let mut sum = Sum::default();
    // Terms past n are zero.
    for i in 0..=k.min(n) {
        sum.add(choose(n, i) * p.powf(i as f64) * (1.0 - p).powf((n - i) as f64));
    }
    sum.value()
}

/// The p in `[0, 1]` where a function that decreases in p crosses `target`.
fn solve(f: impl Fn(f64) -> f64, target: f64) -> f64 {
    let (mut low, mut high) = (0.0f64, 1.0f64);
    for _ in 0..60 {
        let middle = (low + high) / 2.0;
        if f(middle) > target {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low + high) / 2.0
}

/// Exact (Clopper-Pearson) 97.5% lower bound on a failure rate after `k`
/// failures in `n` runs.
pub(super) fn rate_lower(k: u64, n: u64) -> f64 {
    if k == 0 {
        0.0
    } else {
        solve(|p| binom_cdf(k - 1, n, p), CONFIDENCE)
    }
}

/// Exact (Clopper-Pearson) 97.5% upper bound on a failure rate after `k`
/// failures in `n` runs.
pub(super) fn rate_upper(k: u64, n: u64) -> f64 {
    if k >= n {
        1.0
    } else {
        solve(|p| binom_cdf(k, n, p), 1.0 - CONFIDENCE)
    }
}

/// One-sided Fisher exact test: the chance of `f1` or more of all the
/// failures landing on the first side (`n1` runs) when both sides fail
/// equally often. `f0` failures in `n0` runs is the other side.
///
/// NaN when the counts are too large for a finite binomial coefficient
/// (beyond [`MAX_FIXED_RUNS`] a side); a NaN is below no threshold, so it
/// can only read as "not decided".
pub(super) fn fisher_p(f1: u64, n1: u64, f0: u64, n0: u64) -> f64 {
    // Saturating: the counts may come from a report someone else wrote.
    let (total, fails) = (n1.saturating_add(n0), f1.saturating_add(f0));
    let all = choose(total, n1);
    if !all.is_finite() {
        return f64::NAN;
    }
    let mut tail = Sum::default();
    for x in f1..=fails.min(n1) {
        tail.add(choose(fails, x) * choose(total - fails.min(total), n1 - x));
    }
    (tail.value() / all).min(1.0)
}

/// What `--flaky` measured on the two ends, and how sure a verdict on one
/// commit has to be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Failure rate on the good end, smoothed.
    pub p0: f64,
    /// The failure rate a bad commit is judged against: the exact 97.5%
    /// lower bound of the bad end's rate.
    pub p1: f64,
    /// The bad end's rate as measured, smoothed. Not what is tested against.
    pub p1_point_estimate: f64,
    /// Accepted chance of calling a good commit bad.
    pub alpha: f64,
    /// Accepted chance of calling a bad commit good.
    pub beta: f64,
    /// At most this many runs on one commit.
    pub max_runs: u64,
}

impl Stats {
    /// From `(failures, runs)` on each end.
    ///
    /// The good end's rate is smoothed, `(f + 0.5) / (n + 1)`, so it is
    /// never 0 and every run moves the evidence by a finite step.
    ///
    /// The bad end's measured rate is not what commits are judged against.
    /// The measuring stops as soon as the two ends look different, which
    /// favours a lucky, too-high rate on the bad end; a test against that
    /// estimate calls a truly bad commit good about three times more often
    /// than `--alpha` allows (2.6% at alpha 0.01 and a true rate of 30% in
    /// simulation). Judged against the exact 97.5% lower bound of the rate
    /// it is about 0.1%, for roughly twice the runs on a good commit. The
    /// bound is kept at least half as large again as the good end's rate,
    /// so the two hypotheses never coincide.
    pub fn measured(good: (u64, u64), bad: (u64, u64), alpha: f64, max_runs: u64) -> Self {
        let smoothed = |(fails, runs): (u64, u64)| (fails as f64 + 0.5) / (runs as f64 + 1.0);
        let p0 = smoothed(good);
        Self {
            p0,
            p1: rate_lower(bad.0, bad.1).max(p0 * 1.5),
            p1_point_estimate: smoothed(bad),
            alpha,
            beta: alpha,
            max_runs,
        }
    }

    /// Whether a sequential test between the two rates can decide anything:
    /// both must be rates, the bad one the larger. It fails when the good
    /// end itself fails more than two times in three.
    pub fn separates(&self) -> bool {
        self.p0 > 0.0 && self.p1 > self.p0 && self.p1 < 1.0
    }
}

/// Wald's sequential probability ratio test of "fails at the bad end's
/// rate" against "fails at the good end's rate", one run at a time.
#[derive(Debug, Clone)]
pub struct Wald {
    up: f64,
    down: f64,
    on_fail: f64,
    on_pass: f64,
    log_ratio: f64,
}

impl Wald {
    pub fn new(stats: &Stats) -> Self {
        Self {
            up: ((1.0 - stats.beta) / stats.alpha).ln(),
            down: (stats.beta / (1.0 - stats.alpha)).ln(),
            on_fail: (stats.p1 / stats.p0).ln(),
            on_pass: ((1.0 - stats.p1) / (1.0 - stats.p0)).ln(),
            log_ratio: 0.0,
        }
    }

    /// Count one run. `Some("bad")` or `Some("good")` once the evidence
    /// crosses a boundary, `None` while it is still open.
    pub fn observe(&mut self, failed: bool) -> Option<&'static str> {
        self.log_ratio += if failed { self.on_fail } else { self.on_pass };
        if self.log_ratio >= self.up {
            Some("bad")
        } else if self.log_ratio <= self.down {
            Some("good")
        } else {
            None
        }
    }

    /// The log likelihood ratio so far.
    pub fn log_ratio(&self) -> f64 {
        self.log_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equal to the value the Python function printed, to 1e-12 relative.
    fn close(got: f64, want: f64) {
        let scale = want.abs().max(f64::MIN_POSITIVE);
        assert!(
            (got - want).abs() <= 1e-12 * scale,
            "got {got:e}, the reference tool gives {want:e}"
        );
    }

    #[test]
    fn binomial_coefficients_are_exact_where_they_fit_and_close_beyond() {
        assert_eq!(choose(0, 0), 1.0);
        assert_eq!(choose(5, 0), 1.0);
        assert_eq!(choose(5, 5), 1.0);
        assert_eq!(choose(5, 2), 10.0);
        assert_eq!(choose(5, 6), 0.0);
        assert_eq!(choose(50, 25), 126_410_606_437_752.0);
        // float(math.comb(n, k)): the same float while the integer fits 128
        // bits, within the tolerance beyond.
        assert_eq!(choose(100, 50), 1.008913445455642e29);
        assert_eq!(choose(128, 64), 2.3951146041928085e37);
        assert_eq!(choose(130, 65), 9.50676258279607e37);
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(7, 1), 1);
        close(choose(200, 100), 9.054851465610328e58);
        close(choose(509, 170), 2.3934975436435454e139);
        // The largest one a run of MAX_FIXED_RUNS an end can ask for.
        close(choose(1018, 509), 7.022554277884237e304);
    }

    #[test]
    fn the_binomial_cdf_matches_the_reference_tool() {
        for (k, n, p, want) in [
            (0, 10, 0.3, 0.02824752489999998),
            (3, 10, 0.3, 0.6496107183999996),
            (10, 10, 0.3, 0.9999999999999994),
            (5, 50, 0.1, 0.6161230077242777),
            (0, 0, 0.5, 1.0),
            (49, 50, 0.975, 0.7180118976590831),
            (2, 80, 0.01, 0.953446814264068),
            (14, 50, 0.3, 0.4468315742580414),
            // Past 128 runs the coefficients are floats.
            (150, 500, 0.3, 0.5220432311694838),
            (10, 300, 0.05, 0.11230139347749565),
        ] {
            close(binom_cdf(k, n, p), want);
        }
        // Past n there is nothing left to add.
        close(binom_cdf(12, 10, 0.3), binom_cdf(10, 10, 0.3));
        assert_eq!(binom_cdf(3, 10, 0.0), 1.0);
        assert_eq!(binom_cdf(3, 10, 1.0), 0.0);
    }

    #[test]
    fn the_rate_bounds_match_the_reference_tool() {
        for (k, n, want) in [
            (0, 50, 0.0),
            (1, 50, 0.0005062279830408416),
            (15, 50, 0.1786178456641469),
            (50, 50, 0.9288782635358024),
            (3, 10, 0.06673951117773441),
            (30, 100, 0.21240642048953656),
            (170, 500, 0.2985309704329976),
            (1, 1, 0.024999999999999967),
            (0, 0, 0.0),
        ] {
            close(rate_lower(k, n), want);
        }
        for (k, n, want) in [
            (0, 50, 0.07112173646419767),
            (1, 50, 0.10646954571149986),
            (15, 50, 0.4460823256611782),
            (50, 50, 1.0),
            (3, 10, 0.6524528500599973),
            (30, 100, 0.39981467617980415),
            (170, 500, 0.38337423698085427),
            (0, 1, 0.9749999999999999),
            (0, 0, 1.0),
        ] {
            close(rate_upper(k, n), want);
        }
        // The pair brackets the observed rate.
        assert!(rate_lower(15, 50) < 0.3 && 0.3 < rate_upper(15, 50));
        // 15 of 50 with the commit against 0 of 50 without: "at least 2.5
        // times more likely".
        close(rate_lower(15, 50) / rate_upper(0, 50), 2.511438197998192);
    }

    #[test]
    fn the_bisection_finds_where_a_decreasing_function_crosses() {
        let root = solve(|p| 1.0 - p, 0.25);
        assert!((root - 0.75).abs() < 1e-15, "{root}");
        // A function that never gets down to the target ends at 1, one that
        // starts below it ends at 0.
        assert!(solve(|_| 1.0, 0.5) > 1.0 - 1e-15);
        assert!(solve(|_| 0.0, 0.5) < 1e-15);
    }

    #[test]
    fn the_fisher_exact_test_matches_the_reference_tool() {
        for (f1, n1, f0, n0, want) in [
            (3, 10, 0, 10, 0.10526315789473684),
            (5, 10, 0, 10, 0.016253869969040248),
            (7, 20, 0, 20, 0.004158004158004158),
            (15, 50, 0, 50, 8.884673390211095e-06),
            (10, 10, 0, 10, 5.412544112234515e-06),
            (0, 10, 0, 10, 1.0),
            (3, 10, 3, 10, 0.6857585139318886),
            (0, 0, 0, 0, 1.0),
            (2, 5, 0, 0, 1.0),
            (9, 30, 0, 30, 0.0009678016595694545),
            (14, 50, 1, 50, 0.0001939820356862756),
            (0, 50, 15, 50, 1.0),
            (60, 100, 40, 100, 0.0035297577475081232),
            (6, 20, 0, 20, 0.010098010098010098),
            (10, 30, 0, 30, 0.0003985065657050695),
            (5, 15, 0, 15, 0.0210727969348659),
            (20, 60, 0, 60, 1.4227860183065219e-07),
            (33, 100, 0, 100, 4.878688723433145e-12),
            // Past 128 runs in all the coefficients are floats.
            (100, 200, 60, 200, 3.234074178813947e-05),
            (250, 500, 200, 500, 0.0009135038235329013),
            // The largest counts the command accepts: a batch of ten past
            // MAX_FIXED_RUNS on each end.
            (170, 509, 0, 509, 2.6463834847958797e-59),
            (0, 509, 0, 509, 1.0),
        ] {
            close(fisher_p(f1, n1, f0, n0), want);
        }
        // A third of the runs failing on one side passes the 0.001 bar at
        // thirty runs an end, not at twenty.
        assert!(fisher_p(6, 20, 0, 20) >= BASELINE_P);
        assert!(fisher_p(10, 30, 0, 30) < BASELINE_P);
        assert!(fisher_p(10, 30, 0, 30) < REPEATED_P);
        // Beyond the accepted counts it says so instead of calling an
        // overflow significant.
        assert!(fisher_p(1000, 3000, 0, 3000).is_nan());
    }

    /// The rates of the reference tool's version 7 trace: 0 of 20 on the
    /// good end, 7 of 20 on the bad end, judged against the point estimate.
    fn stats() -> Stats {
        let measured = Stats::measured((0, 20), (7, 20), 0.01, 80);
        Stats {
            p1: measured.p1_point_estimate,
            ..measured
        }
    }

    /// Feed runs to the test the way the measuring loop does.
    fn decide(runs: impl Iterator<Item = bool>, max_runs: u64) -> (&'static str, u64, u64, f64) {
        let mut wald = Wald::new(&stats());
        let (mut count, mut fails) = (0, 0);
        for failed in runs {
            if count >= max_runs {
                break;
            }
            count += 1;
            fails += u64::from(failed);
            if let Some(verdict) = wald.observe(failed) {
                return (verdict, count, fails, wald.log_ratio());
            }
        }
        ("skip", count, fails, wald.log_ratio())
    }

    #[test]
    fn a_bad_commit_is_judged_against_the_lower_bound_of_the_measured_rate() {
        // walk-verify.py 8, flaky_baseline: 0 of 20 against 9 of 20.
        let stats = Stats::measured((0, 20), (9, 20), 0.01, 200);
        close(stats.p0, 0.023809523809523808);
        close(stats.p1, 0.23057789677592416);
        close(stats.p1_point_estimate, 0.4523809523809524);
        assert_eq!((stats.alpha, stats.beta, stats.max_runs), (0.01, 0.01, 200));
        assert!(stats.separates());
        // 0 of 30 against 10 of 30, and 0 of 20 against 7 of 20.
        let stats = Stats::measured((0, 30), (10, 30), 0.01, 200);
        close(stats.p0, 0.016129032258064516);
        close(stats.p1, 0.1728742215260391);
        close(stats.p1_point_estimate, 0.3387096774193548);
        close(
            Stats::measured((0, 20), (7, 20), 0.01, 200).p1,
            0.153909204784541,
        );
        close(
            Stats::measured((0, 10), (5, 10), 0.01, 80).p0,
            0.045454545454545456,
        );
        // A bound that would fall to the good end's rate is held at one and
        // a half times that rate.
        let close_ends = Stats::measured((5, 10), (6, 10), 0.01, 80);
        close(close_ends.p1, close_ends.p0 * 1.5);
        assert!(close_ends.separates());
        // A good end that fails most of the time leaves no room above it.
        let failing_good = Stats::measured((70, 100), (100, 100), 0.01, 80);
        assert!(failing_good.p1 >= 1.0);
        assert!(!failing_good.separates());
    }

    /// Runs of the sequential test the point estimate gets wrong and the
    /// lower bound gets right: the bad end failed 9 of 20, and a commit
    /// that really fails 3 runs in 13 passes its first 10.
    #[test]
    fn the_lower_bound_does_not_call_a_slow_failure_good() {
        let measured = Stats::measured((0, 20), (9, 20), 0.01, 200);
        let ten_pass_three_fail = |run: u64| run % 13 >= 10;
        let decide = |stats: &Stats| {
            let mut wald = Wald::new(stats);
            let (mut fails, mut runs) = (0u64, 0u64);
            loop {
                let failed = ten_pass_three_fail(runs);
                runs += 1;
                fails += u64::from(failed);
                if let Some(verdict) = wald.observe(failed) {
                    return (verdict, runs, fails);
                }
            }
        };
        let against_point_estimate = Stats {
            p1: measured.p1_point_estimate,
            ..measured
        };
        assert_eq!(decide(&against_point_estimate), ("good", 8, 0));
        assert_eq!(decide(&measured), ("bad", 25, 5));
        // A commit that never fails needs 20 passes now, not 8.
        let mut wald = Wald::new(&measured);
        let passes = (1..).find(|_| wald.observe(false).is_some()).unwrap();
        assert_eq!(passes, 20);
    }

    #[test]
    fn the_sequential_test_matches_the_reference_tool_run_for_run() {
        let wald = Wald::new(&stats());
        close(wald.up, 4.59511985013459);
        close(wald.down, -4.59511985013459);
        close(wald.on_fail, 2.70805020110221);
        close(wald.on_pass, -0.4177352006999788);

        let (verdict, runs, fails, log_ratio) = decide(std::iter::repeat(false), 80);
        assert_eq!((verdict, runs, fails), ("good", 12, 0));
        close(log_ratio, -5.012822408399746);

        let (verdict, runs, fails, log_ratio) = decide(std::iter::repeat(true), 80);
        assert_eq!((verdict, runs, fails), ("bad", 2, 2));
        close(log_ratio, 5.41610040220442);

        // Fails on every third run.
        let (verdict, runs, fails, log_ratio) = decide((0..).map(|run| run % 3 == 2), 80);
        assert_eq!((verdict, runs, fails), ("bad", 9, 3));
        close(log_ratio, 5.617739399106757);

        // One early failure, then it keeps passing.
        let early = std::iter::once(true).chain(std::iter::repeat(false));
        let (verdict, runs, fails, log_ratio) = decide(early, 80);
        assert_eq!((verdict, runs, fails), ("good", 19, 1));
        close(log_ratio, -4.811183411497409);

        // Out of runs before the evidence is decisive: cannot test.
        let (verdict, runs, fails, log_ratio) = decide(std::iter::repeat(false), 5);
        assert_eq!((verdict, runs, fails), ("skip", 5, 0));
        close(log_ratio, -2.088676003499894);
    }
}
