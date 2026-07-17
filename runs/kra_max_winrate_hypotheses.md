# KRA maximum Top-1 research ledger

## Locked promotion contract

- Baseline: current production v4 intrinsic pre-race ranking.
- Primary metric: full-coverage race Top-1 accuracy.
- Promotion floor: absolute OOS lift at least `+5.0%p`.
- Confirmation: every chronological fold positive, paired bootstrap 95% lower bound positive, Top-3 and log-loss non-inferior.
- Final guard: candidate selection must not use the post-2026-06-22 fresh holdout.

## H1: leakage-safe dynamic performance

Pre-registered before execution. Add only historical, already-completed-race measurements: prior speed, recent speed, early-position tendency, finish gain, body-weight change, burden change, and distance change. Evaluate winner, place, finish-rank, and restricted ensemble score families on the existing chronological folds. Same-race time, sectional position, and finish are never exposed to that race's prediction row.

Rationale: v4 uses cumulative horse outcomes but omits dynamic covariates from completed races. Covariate-assisted Plackett-Luce research supports combining time-varying covariates with participant effects for ranking, while the local result schema provides actual historical times and positions.

Result: FAIL. Selected `dynamic_extra_d12@0.75` reached pooled historical `+0.843%p` with CI `+0.123..+1.562%p`, but fresh holdout lift was `0.000%p`, Top-3 fell `2.239%p`, and log-loss worsened. No promotion.

## H2: context-adjusted dynamic strength

Pre-registered before execution. Extend H1 with completed-race speed relative to that race's median, same-distance horse form, rolling recent jockey/trainer form, current class, track moisture, month, meet, and weather context. Keep candidate selection limited to the three discovery folds and preserve the same confirmation and promotion gates. H1 report remains frozen at `runs/kra_max_winrate_results.json`; H2 writes a separate report.

Result: FAIL. Selected `dynamic_win_d4@0.75` reached pooled historical `+1.131%p` with CI `+0.226..+2.055%p`; fresh holdout was `+1.493%p` with CI `-4.478..+7.463%p`. Direction was positive in every fold and Top-3/log-loss improved, but the absolute `+5.0%p` gate was not met.

## H3: same-day-safe dynamic Elo strength

Pre-registered before execution. Add horse, jockey, and trainer Elo-style ratings updated from full finishing order only after an entire race date has completed. All participants on the same date receive the pre-date rating, preventing earlier races that day from leaking into later races. Combine these relative ratings with H2 context features and rerun the locked model/weight grid without changing the selection rule.

Result: FAIL. Selected `dynamic_win_d2@1.00` reached pooled historical `+1.521%p` with CI `+0.493..+2.590%p`; fresh holdout was `+1.493%p` with CI `-5.224..+8.209%p`, while fresh Top-3 regressed. The absolute gate was not met.

## H4: rich pairwise learning-to-rank

Pre-registered before execution. Add a winner-versus-loser HGB pairwise ranker using the full H3 feature set, normalize its race scores, and evaluate the same fixed blend weights. This tests whether direct relative optimization captures field composition better than per-horse winner classification without changing folds or promotion gates.

Result: FAIL. The rich pairwise family did not beat the H3 selected classifier and worsened probability loss at larger weights. The selected report remained H3's `+1.521%p` historical and `+1.493%p` fresh result.

## H5: time-safe pedigree offspring strength

Pre-registered before execution. Join only static sire and dam identifiers from the horse registry, then compute each parent's offspring win, place, and finish priors using races from strictly earlier dates. Do not use registry cumulative race/win counts. Add pedigree-relative features to H4 and retain the exact same model, fold, and promotion contracts.

Result: FAIL. Selected rich pairwise `@0.75` reached historical `+1.213%p` with CI `+0.514..+1.932%p`, but fresh holdout fell `-0.746%p`, with Top-3 and log-loss regressions. The best research candidate remains H3, which is still below the hard gate and is not enabled in production.

## H6: official as-of entry-sheet form

Pre-registered before execution. Parse archived KRA entry sheets at their original publication time and add career, recent-one-year, and recent-six-month wins, places, starts, and earnings. Join strictly by meet, race date, race number, and saddlecloth number. Preserve the fixed discovery folds and require a new post-2026-07-11 forward holdout before any promotion.

Result: FAIL. The archive covered every evaluated runner. Selected rich pairwise `@1.00` reached historical `+1.459%p` with CI `+0.514..+2.405%p`, but log-loss worsened and the already-observed holdout fell `-1.493%p`. The absolute gate was not met.

## H7: official pre-race conditioning and health

Pre-registered before execution. Extract only fields printed in each archived pre-race program before the race: training session count and minutes, canter and gallop counts, swimming sessions and laps, recent medical-treatment frequency and recency, and starting-training recency/result. Normalize within race and combine with v4 without using results printed elsewhere on the card. Select on the same three discovery folds, confirm chronologically, and reserve races after 2026-07-11 for the required untouched forward guard.

Result: FAIL for the first independently testable conditioning slice. Two-week session and minute features covered every runner. Selected place model `@0.50` improved historical Top-1 `+1.131%p` with CI `+0.514..+1.768%p` and improved Top-3/log-loss, but the already-observed holdout fell `-1.493%p`. No promotion.

## H8: strictly pre-date veterinary health

Pre-registered before execution. Parse each archived race's medical page, exclude every treatment dated on or after the race date, and derive recent treatment, locomotor injury, respiratory condition, fatigue-treatment, vaccination, and recency features. This strict cutoff avoids same-day post-race leakage. Combine with the time-safe H3 rating/context set, select only on the three discovery folds, and retain the post-2026-07-11 forward guard.

Result: FAIL. Veterinary coverage was 100%. Selected winner HGB `dynamic_win_d3@1.00` improved every chronological fold, pooled historical Top-1 `+1.151%p` with CI `+0.123..+2.220%p`, Top-3, and log-loss. The already-observed holdout improved `+2.239%p`, but its CI crossed zero and the absolute `+5.0%p` gate was not met. No promotion.

## H9: diversified race-list ranking and segmented experts

Pre-registered before execution. Replace the repeated fixed HGB blend grid with independently structured candidates: listwise race-softmax optimization, race-context segmented experts with global fallback, OOF stacking, and pace/field interaction features. Keep all features as-of the race date, tune only inside chronological discovery folds, and evaluate the untouched confirmation fold without retuning. The already-observed 2026-06-22..2026-07-11 period remains diagnostic only; promotion still requires post-2026-07-11 results.

Result: FAIL. The diversified frame used 165 features and the selected closing-market probability teacher, trained only from historical races and never reading current-race odds, improved pooled historical Top-1 `+2.425%p` with CI `+1.316..+3.535%p`. It improved every fold, Top-3, and log-loss, but the already-observed holdout improved only `+0.746%p`. The separate live closing-odds ceiling was `+8.209%p` with CI `+0.746..+16.418%p`, but it is not eligible for intrinsic promotion because a timestamped pre-start odds snapshot is required.

## H10: diversified historical market-teacher targets

Pre-registered before execution. Train separate pre-race models to imitate historical closing market probability, market favorite identity, place-market probability, and market-favorite listwise order while excluding all current-race odds from prediction inputs. Add segmented favorite experts and fixed consensus ensembles. Preserve the H9 feature frame, discovery-only selection, confirmation fold, and post-2026-07-11 forward guard.

Result: FAIL. Discovery selection again chose the historical market-probability teacher at pooled `+2.425%p`; the selected candidate improved the already-observed holdout only `+0.746%p`. A non-selected segmented market-favorite expert produced diagnostic holdout `+6.716%p`, but its historical pooled lift was only about `+1.57%p`, its probability loss worsened, and its holdout CI lower bound was `0.0%p`. It is therefore a lead for prospective testing, not promotable evidence.

## H11: nested chronological model-routing gate

Pre-registered before execution. For every outer chronological fold, train diverse base rankers on an earlier inner block, generate strictly later gate-training predictions, learn a race-level selector from candidate confidence, agreement, field, distance, rating spread, and cross-model pick probabilities, then refit base rankers on the full outer training history. Evaluate hard, soft, and conservative gates on the untouched outer fold. Current-race odds remain excluded.

Result: FAIL. The selected soft gate `@0.50` improved all four outer folds and pooled historical Top-1 `+1.459%p` with CI `+0.658..+2.282%p`, while improving Top-3 and log-loss. The already-observed holdout improved `+2.239%p`, but Top-3 fell and its CI crossed zero. No promotion.

## H12: official pre-start live market gate

Pre-registered before runtime enablement. Treat historical closing odds only as a retrospective ceiling, never as timestamp proof. Enable the existing live market rank only when the current request has a complete positive odds board, no published result, an official KRA weekly start time, and a KST fetch timestamp strictly before that start. Keep the intrinsic model unchanged when any condition is missing. Require prospective timestamped post-2026-07-11 snapshots before promotion claims.

Result: IMPLEMENTED, FORWARD AUDIT PENDING. The retrospective closing-market ceiling improved all four chronological folds by `+8.139..+10.336%p`; pooled Top-1 improved `+9.044%p` with CI `+7.708..+10.380%p`, while Top-3 and log-loss also improved. The already-observed holdout improved `+8.209%p` with CI `+0.746..+16.418%p`. Runtime activation now requires official KRA start-time verification and a strictly pre-start complete board with no result fields. Because historical result rows do not prove when odds were captured, promotion remains false until timestamped forward snapshots reproduce the lift.

## H13: market residual, heterogeneous rankers, restricted rerank, and uncertainty gates

Pre-registered before execution. Change the benchmark from intrinsic v4 to the stronger normalized win market and allow current-race odds only in this explicitly live-only retrospective experiment. Add 12 race-relative market features, winner HGB, market residual HGB, compact conditional logit, Extra Trees, Random Forest, logistic regression, consensus models, linear blends, market-top-2/3 restricted reranking, and favorite-probability/gap uncertainty gates. Select 240 policies on the first three chronological folds and lock the fourth fold for confirmation.

Result: FAIL. The best discovery policy was `market_extra_trees|top2_uncertainty|0.30|0.03`. Its fold lifts versus the pure market were `+0.246`, `+0.483`, `0.000`, and `-0.084%p`; pooled lift was only `+0.164%p` with bootstrap CI `-0.308..+0.637%p`. The already-observed holdout lift was `0.000%p`. No production connection.

## H14: race-level favorite-versus-challenger switching

Pre-registered before execution. Reframe Top-1 correction as a race-level multiclass decision: retain the market favorite or swap to market rank 2/3. Use 117 market-ranked runner strength features, three class-balance powers, two candidate depths, and five conservative switching margins. Choose on the first three folds and evaluate the fourth without retuning.

Result: FAIL. The selected top-2 challenger with class-balance power `0.25` and margin `0.15` produced fold lifts `+0.328`, `+0.564`, `-0.411`, and `-0.084%p`. Pooled lift was `+0.103%p` with CI `-0.411..+0.617%p`; log-loss slightly worsened. The already-observed holdout improved `+2.239%p`, but it is reused, below `+5.0%p`, and not historically stable. No promotion.

## H15: timestamped odds-trajectory data acquisition

Pre-registered after H13/H14 confirmed that final-board static corrections do not reliably beat the market. The local application database contained `0` KRA market snapshots, so odds-movement models could not be evaluated without fabricating timing evidence. Persist each officially verified pre-start complete KRA board encountered by the prediction routes with capture timestamp, deduplicate unchanged boards inside one minute, and retain changed boards immediately. This creates the forward-safe dataset required for future trajectory features such as probability velocity, rank changes, late steam/drift, and market disagreement.

Result: IMPLEMENTED, DATA ACCUMULATION PENDING. SQLite and PostgreSQL persistence paths are covered, prediction remains fail-open if telemetry storage fails, and no trajectory model is promoted until multiple timestamped boards per race plus settled outcomes exist.

## H16: venue-aware sectional efficiency

Pre-registered before execution. Decode Seoul and Busan cumulative G3F fields into final-600m time while using Jeju's direct G3F time, then derive strictly historical first-200m speed, final-600m speed, finishing-speed percentage, early/late race-relative advantage, pace balance, consistency, and four-race rolling form. Exclude current-race odds and current-race sectionals from every prediction row.

Result: FAIL. Selected place classifier `dynamic_place_d3@0.50` improved pooled historical Top-1 `+0.822%p` with CI `+0.206..+1.439%p`, but the reused latest guard fell `-0.746%p`, Top-3 and log-loss regressed, and the candidate trailed retrospective closing-market Top-1 by `7.40..9.60%p` across folds. No production connection.

## H17: world-model family benchmark on sectional fundamentals

Pre-registered before execution. Compare winner-only linear SVM classification, multilayer neural networks, Random Forests, and race-list conditional logit on the same odds-free sectional and v4 feature frame. Select only on the first three chronological folds, preserve the fourth confirmation fold, and compare each selected score with both v4 and the retrospective closing market.

Result: FAIL. Selected `conditional_logit_r2@0.25` improved pooled historical Top-1 `+1.131%p` with CI `+0.555..+1.706%p` and improved all four folds, but the reused latest guard was `0.000%p`, Top-3 regressed, and the candidate trailed retrospective closing-market Top-1 by `7.17..9.19%p`. SVM, neural-network, and Random-Forest candidates did not clear the locked `+5.0%p` gate. No production connection.

## H18: wide greedy drug-discovery ensemble

Pre-registered before execution. Treat each convex ensemble as a molecular genome over 16 independently structured base rankers, cap current-race market influence at `15%`, and assay at least `20,000` genomes for each of five generations per cycle. Seed every base model, then diversify through sparse and dense Dirichlet sampling, mutation, and crossover. Select the beam only on the first three chronological discovery folds; the fourth confirmation fold must never influence frontier selection. Continue statefully until every fold is within `0.5%p` of the retrospective market and at least `+5.0%p` above v4, then require a new pristine forward holdout before promotion.

Result after three broad cycles: SEARCHING, NOT PROMOTED. `300,000` genomes were assayed. The best genome used `14.36%` market weight and improved v4 by `+9.19`, `+7.09`, `+7.81`, and `+6.90%p`, but remained behind the retrospective market by `-1.15`, `-1.05`, `-1.07`, and `-1.93%p`. The confirmation bootstrap CI versus market was entirely negative (`-3.45..-0.50%p`). Durable state, frontier, prediction cache, and append-only cycle ledger were saved; production remains unchanged.

## H19: full-finishing-order ranking expansion

Pre-registered before execution. Expand the drug-discovery library from 16 to 20 base models with two Plackett-Luce models trained on the observed finishing sequence and two full-order pairwise HGB rankers trained on every observed precedence pair. Preserve the 15% direct market cap, discovery-only frontier selection, and untouched confirmation fold.

Result: FAIL. The four new rankers were individually `6.39..9.42%p` below the retrospective market, received no weight in the selected genome, and the refreshed 100,000-genome cycle remained `-1.15`, `-1.13`, `-1.15`, and `-2.02%p` below market. No production connection.

## H20: pairwise market-favorite error correction

Pre-registered before execution. Train a binary pairwise challenger model over market top-2/top-3 runners using 117 independent runner-strength and difference features. Search class balancing and conservative switch thresholds while selecting only on the three discovery folds. Preserve market probabilities except when the learned challenger clears the locked threshold.

Result: FAIL. Across 152 policies, the selected rule switched only `0.16..0.32%` of races and improved pooled Top-1 by `+0.041%p`, but its bootstrap CI was `-0.062..+0.144%p`, confirmation lift was `0.0%p`, and the reused latest holdout was unchanged. No promotion.

## H21: context-gated pairwise market correction

Pre-registered before execution. Expand H20 across 28 venue, distance, favorite-uncertainty, and interaction gates, producing 4,256 policies. Selection remains discovery-only; confirmation is reported only after the policy is locked. A separate diagnostic may identify all-fold-positive policies, but any policy discovered using confirmation is explicitly post-selection and requires a new forward period.

Result: FAIL for promotion; one forward watchlist lead found. The discovery-selected rule improved the three discovery folds by `+0.410`, `+0.322`, and `+0.164%p`, but confirmation fell `-0.336%p`. A separate post-selection Jeju/high-uncertainty top-2 rule was positive on all four historical folds by `+0.082..+0.164%p`, but pooled CI crossed zero (`-0.082..+0.329%p`) and the reused latest holdout was unchanged. Because confirmation was used to identify it, this is not OOS proof and is not connected to production.

## H22: strict market-plus-five gate with enriched as-of feature library

The acceptance contract was corrected to require at least `+5.0%p` Top-1 versus the retrospective market on every discovery and confirmation fold, in addition to the existing `+5.0%p` v4 lift. The drug-discovery library was expanded with time-safe dynamic ratings/context, recent-form, distance/venue/jockey-pair priors, owner/tool features, time-safe pedigree priors, and sectional history. The direct current-race market contribution remained capped at `15%`; selection remained limited to the first three chronological folds.

Result: FAIL, correctly blocked. Cycle 33 assayed `100,000` new genomes (`3,300,000` cumulative) across 20 models. The best enriched genome improved v4 by `+9.024`, `+6.849`, and `+7.566%p` on discovery and `+6.728%p` on confirmation, but trailed the market by `-1.313`, `-1.289`, `-1.316`, and `-2.103%p`. The new strict `+5.0%p` market gate stayed false; no model was promoted or connected to production.

## H23: structured hybrid genome search

Replace part of the random Dirichlet budget with deterministic pairwise model mixtures at five weight ratios and three market-boundary values, then retain parent mutation/crossover and stochastic exploration. This tests whether the stagnation came from missing low-order ensemble boundaries rather than insufficient candidate count. The OOS folds, market cap, and strict `+5.0%p` market gate remain unchanged.

Result: FAIL, no search-efficiency gain yet. A separate 100,000-genome run with a cache hit selected market gaps `-1.313`, `-1.209`, `-1.151`, and confirmation `-2.103%p`; the preceding wide run selected `-1.231`, `-1.209`, `-1.234`, and `-2.103%p`. The hybrid method improved one fold but worsened the worst fold and did not approach the required `+5.0%p`. It is retained as an exploratory generator, not a promotion path.

## H24: multi-fidelity Pareto-diversity beam search

Replace full-fold scoring of every genome with three discovery stages: score all candidates on fold 1, retain a Pareto-diverse beam; rescore that beam on folds 1-2, retain a smaller Pareto-diverse beam; rescore the final beam on folds 1-3, then run the locked confirmation fold only on the survivors. The Pareto objectives are worst and mean market gap plus worst and mean v4 lift; a weight-space max-min distance term prevents collapse onto one nearly identical ensemble. The hybrid structured generator, 15% market cap, chronological folds, and strict `+5.0%p` market gate remain unchanged.

Result: FAIL, architecture improved but no promotion edge. The separate eight-generation run screened `160,000` candidate genomes with `165,632` staged assay evaluations and a prediction-cache hit. Its selected discovery market gaps were `-1.066`, `-1.209`, and `-1.480%p`; confirmation was `-2.019%p`. v4 lifts remained `+9.270`, `+6.930`, `+7.401`, and `+6.812%p`, but every market gap remained negative and the strict `+5.0%p` gate stayed false. This is retained as the new search architecture; no candidate is connected to production.
