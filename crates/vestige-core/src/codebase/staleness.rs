//! Staleness prediction: infer when a code memory has probably gone stale
//! instead of waiting to be told.
//!
//! The estimator is backcalculation in the Brookmeyer & Gail (1986) sense:
//! staleness is only ever OBSERVED at verification time, some unknown lag
//! after the code actually moved. Each verification is therefore one
//! observation about the capture-to-rot lag distribution:
//!
//! * a verification that found drift at age `a` says rot happened at or
//!   before `a` (an event, interval-censored at the verification age);
//! * a verification that found the code intact at age `a` says rot had not
//!   happened by `a` (a right-censored observation).
//!
//! From those two kinds of evidence a Kaplan-Meier product-limit estimator
//! recovers the survival curve S(t) = P(still fresh at age t), and
//! P(stale by age t) = 1 - S(t) becomes the prediction for a memory of age
//! `t` that has no fresh verification of its own.
//!
//! Honesty rules, in order of importance:
//!
//! 1. **No prediction without evidence.** Below [`MIN_OBSERVATIONS`] total
//!    observations and [`MIN_EVENTS`] actual drift detections, `fit` returns
//!    `None` and callers must say "insufficient verification history", never
//!    a number. A brand-new V31 table predicts nothing.
//! 2. **Unverifiable is not evidence.** An anchor whose file cannot be read
//!    contributes nothing in either direction.
//! 3. **A prediction is a probability, not an accusation.** Callers surface
//!    it as `predictedStaleProbability` next to the memory; nothing is
//!    demoted, suppressed, or deleted on the strength of a prediction.

/// Total (event + censored) observations required before fitting.
pub const MIN_OBSERVATIONS: usize = 12;

/// Observed drift detections required before fitting. A history of only
/// "still fresh" checks cannot locate the onset distribution at all — the
/// mirror image of the success-only lesson from the decay optimizer.
pub const MIN_EVENTS: usize = 4;

/// One verification outcome, reduced to what the estimator needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StalenessObservation {
    /// Age of the memory at verification time, in days.
    pub age_days: f64,
    /// True if the verification found drift (an event); false if the code
    /// was intact (right-censored).
    pub drifted: bool,
}

/// Fitted product-limit estimator over verification history.
#[derive(Debug, Clone)]
pub struct StalenessPredictor {
    /// (age_days, survival) steps, age ascending, survival non-increasing.
    steps: Vec<(f64, f64)>,
    events: usize,
    observations: usize,
}

impl StalenessPredictor {
    /// Fit from verification history. Returns `None` when the evidence is
    /// insufficient to say anything honest.
    pub fn fit(observations: &[StalenessObservation]) -> Option<Self> {
        let usable: Vec<StalenessObservation> = observations
            .iter()
            .copied()
            .filter(|obs| obs.age_days.is_finite() && obs.age_days >= 0.0)
            .collect();
        let events = usable.iter().filter(|obs| obs.drifted).count();
        if usable.len() < MIN_OBSERVATIONS || events < MIN_EVENTS {
            return None;
        }

        let mut sorted = usable.clone();
        sorted.sort_by(|a, b| a.age_days.partial_cmp(&b.age_days).unwrap());

        // Kaplan-Meier product-limit: at each distinct event age t with d
        // events and n at risk (age >= t), survival *= (1 - d/n).
        let mut steps = Vec::new();
        let mut survival = 1.0_f64;
        let total = sorted.len();
        let mut index = 0;
        while index < sorted.len() {
            let age = sorted[index].age_days;
            let mut events_here = 0usize;
            let mut group_end = index;
            while group_end < sorted.len() && sorted[group_end].age_days == age {
                if sorted[group_end].drifted {
                    events_here += 1;
                }
                group_end += 1;
            }
            if events_here > 0 {
                let at_risk = total - index;
                survival *= 1.0 - (events_here as f64 / at_risk as f64);
                steps.push((age, survival.max(0.0)));
            }
            index = group_end;
        }

        Some(Self {
            steps,
            events,
            observations: total,
        })
    }

    /// P(stale by `age_days`), stepwise from the fitted survival curve.
    ///
    /// Beyond the last observed event age the curve is flat: the estimator
    /// refuses to extrapolate a trend it never saw, so a very old memory
    /// gets the last observed probability, not an invented certainty.
    pub fn predict_stale_probability(&self, age_days: f64) -> f64 {
        let mut survival = 1.0;
        for (age, step_survival) in &self.steps {
            if age_days >= *age {
                survival = *step_survival;
            } else {
                break;
            }
        }
        (1.0 - survival).clamp(0.0, 1.0)
    }

    pub fn events(&self) -> usize {
        self.events
    }

    pub fn observations(&self) -> usize {
        self.observations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(age_days: f64, drifted: bool) -> StalenessObservation {
        StalenessObservation { age_days, drifted }
    }

    #[test]
    fn refuses_to_fit_without_enough_observations() {
        let few: Vec<_> = (0..MIN_OBSERVATIONS - 1)
            .map(|i| obs(i as f64, i % 2 == 0))
            .collect();
        assert!(StalenessPredictor::fit(&few).is_none());
    }

    /// The mirror of the decay-optimizer lesson: a history of only
    /// "still fresh" checks must refuse to fit, not fit "nothing rots".
    #[test]
    fn refuses_to_fit_from_fresh_only_history() {
        let fresh_only: Vec<_> = (0..40).map(|i| obs(i as f64, false)).collect();
        assert!(StalenessPredictor::fit(&fresh_only).is_none());
    }

    #[test]
    fn probability_rises_with_age_and_respects_censoring() {
        // 8 drift detections spread over ages, 8 fresh checks censoring
        // the young end.
        let mut history = Vec::new();
        for age in [30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0] {
            history.push(obs(age, true));
        }
        for age in [10.0, 20.0, 40.0, 50.0, 70.0, 80.0, 100.0, 110.0] {
            history.push(obs(age, false));
        }
        let predictor = StalenessPredictor::fit(&history).expect("enough evidence");

        let young = predictor.predict_stale_probability(5.0);
        let mid = predictor.predict_stale_probability(100.0);
        let old = predictor.predict_stale_probability(250.0);
        assert_eq!(young, 0.0, "no events before age 30, nothing to predict");
        assert!(mid > young && old > mid, "monotone in age");
        // The largest observation here is an event, so the product-limit
        // curve legitimately reaches 1.0 at the far end — standard KM.
        assert!(old <= 1.0);

        // Beyond the data the curve is flat, never extrapolated.
        assert_eq!(
            predictor.predict_stale_probability(10_000.0),
            predictor.predict_stale_probability(250.0)
        );
    }

    /// Kaplan-Meier hand check: 4 events among 6 still at risk at age 100
    /// gives survival 1 - 4/6 = 1/3, so P(stale by 100) = 2/3.
    #[test]
    fn product_limit_matches_hand_computation() {
        let mut history = Vec::new();
        for age in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0] {
            history.push(obs(age, false));
        }
        for _ in 0..4 {
            history.push(obs(100.0, true));
        }
        history.push(obs(100.0, false));
        history.push(obs(100.0, false));
        let predictor = StalenessPredictor::fit(&history).expect("12 obs, 4 events");
        let predicted = predictor.predict_stale_probability(120.0);
        assert!((predicted - 2.0 / 3.0).abs() < 1e-9, "got {predicted}");
        assert_eq!(predictor.predict_stale_probability(99.0), 0.0);
    }
}
