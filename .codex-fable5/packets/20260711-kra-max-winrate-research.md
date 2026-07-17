# Fable5 audit packet — KRA maximum Top-1 research

## Verdict requested

`PASS`, `NEEDS_FIX`, `FAIL` 중 하나로 판정한다. 이번 판정은 후보 승격이 아니라, `+5.0%p` 미달 후보를 production에서 차단한 연구 절차와 결론의 정직성을 감사한다.

## Locked contract

- 현재 production v4 intrinsic Top-1을 직접 기준으로 사용한다.
- discovery는 2024H2, 2025H1, 2025H2 세 구간만 사용한다.
- 2026-01-01..2026-06-21은 confirmation, 2026-06-22 이후 134경주는 fresh guard다.
- full-coverage Top-1 절대 `+5.0%p`, 모든 fold 양수, bootstrap CI 하한 양수, Top-3와 log-loss 비악화가 모두 필요하다.

## Experiments

- H1 dynamic speed/form: historical `+0.843%p`, fresh `0.000%p`, FAIL.
- H2 context-adjusted strength: historical `+1.131%p`, fresh `+1.493%p`, FAIL.
- H3 same-day-safe Elo: historical `+1.521%p`, fresh `+1.493%p`, fresh Top-3 regression, FAIL.
- H4 rich pairwise: H3를 넘지 못함, FAIL.
- H5 time-safe pedigree: historical `+1.213%p`, fresh `-0.746%p`, FAIL.
- H6 official archived entry sheet: historical `+1.459%p`, fresh `-1.493%p`, FAIL.
- H7 official two-week conditioning: historical `+1.131%p`, fresh `-1.493%p`, FAIL.
- H8 strictly pre-date veterinary health: historical `+1.151%p`, fresh `+2.239%p`, FAIL.
- H9 listwise/segmented/pace/market-distillation ensemble: historical `+2.425%p`, fresh `+0.746%p`, FAIL.
- H10 diversified historical market targets: selected result identical to H9. A non-selected segmented favorite reached diagnostic fresh `+6.716%p`, but historical lift was about `+1.57%p`, log-loss worsened, and the fresh CI lower bound was `0.0%p`; FAIL.
- H11 nested chronological model gate: historical `+1.459%p`, fresh `+2.239%p`, fresh Top-3 regression, FAIL.
- H12 retrospective closing-market ceiling: all four folds `+8.139..+10.336%p`; pooled `+9.044%p` with CI `+7.708..+10.380%p`; observed holdout `+8.209%p` with CI `+0.746..+16.418%p`. This is not promotion evidence because historical result rows do not prove pre-start capture time.
- H13 live-only market residual/static correction: 240 policies across HGB residuals, conditional logit, Extra Trees, Random Forest, logistic regression, blends, restricted reranks, and uncertainty gates. Best pooled lift versus pure market `+0.164%p`, CI `-0.308..+0.637%p`; confirmation `-0.084%p`; FAIL.
- H14 race-level favorite/challenger switch: 30 multiclass policies. Best pooled lift versus pure market `+0.103%p`, CI `-0.411..+0.617%p`; confirmation `-0.084%p`; observed reused holdout `+2.239%p`; FAIL.
- H15 odds trajectory prerequisite: local DB had zero KRA market snapshots. Officially verified pre-start KRA boards encountered by prediction routes are now persisted with timestamps, unchanged-board one-minute deduplication, and fail-open telemetry handling. Data accumulation pending; no model promotion.
- H16 venue-aware sectional efficiency: historical `+0.822%p`, CI `+0.206..+1.439%p`; reused latest guard `-0.746%p`; retrospective market gap `-7.40..-9.60%p`; FAIL.
- H17 winner SVM, neural network, Random Forest, and conditional-logit benchmark: selected conditional logit historical `+1.131%p`, CI `+0.555..+1.706%p`; reused latest guard `0.000%p`; retrospective market gap `-7.17..-9.19%p`; FAIL.
- H18 wide greedy drug-discovery ensemble: three cycles, five generations per cycle, `300,000` genomes total, 16-model library, and market weight capped at `15%`. Best fold lifts versus v4 were `+9.19`, `+7.09`, `+7.81`, and `+6.90%p`; gaps versus retrospective market remained `-1.15`, `-1.05`, `-1.07`, and `-1.93%p`. Confirmation CI versus market was `-3.45..-0.50%p`; SEARCHING, NOT PROMOTED.
- H19 full-order Plackett-Luce and all-pairs HGB expansion: four new rankers and one refreshed 100,000-genome cycle; selected market gaps `-1.15`, `-1.13`, `-1.15`, and `-2.02%p`; FAIL.
- H20 pairwise market-favorite correction: 152 policies, pooled market lift `+0.041%p`, CI `-0.062..+0.144%p`, confirmation and reused holdout `0.0%p`; FAIL.
- H21 context-gated pairwise correction: 4,256 policies. Discovery-selected confirmation was `-0.336%p`. One post-selection rule was positive on all four historical folds by `+0.082..+0.164%p`, but pooled CI was `-0.082..+0.329%p` and reused holdout was unchanged; WATCHLIST ONLY, NOT PROMOTED.
- H22 strict market-plus-five gate and enriched as-of library: acceptance now requires `>=+5.0%p` versus the retrospective market on every discovery and confirmation fold, not merely parity. Time-safe dynamic ratings/context, recent-form, distance/venue/jockey-pair priors, owner/tool features, pedigree priors, and sectional history were added to the 20-model library. Cycle 33 assayed `100,000` genomes (`3,300,000` cumulative); best market gaps were `-1.313`, `-1.289`, `-1.316`, and `-2.103%p`, while v4 lifts were `+9.024`, `+6.849`, `+7.566`, and `+6.728%p`. FAIL, correctly blocked.
- H23 structured hybrid genome search: deterministic pairwise mixtures at five ratios and three market boundaries were added before mutation/crossover. A separate 100,000-genome cache-hit run selected market gaps `-1.313`, `-1.209`, `-1.151`, and confirmation `-2.103%p`, versus the preceding wide run `-1.231`, `-1.209`, `-1.234`, and `-2.103%p`. FAIL; no promotion or efficiency claim.
- H24 multi-fidelity Pareto-diversity beam search: all genomes are screened on fold 1, then smaller Pareto/diversity beams are screened on folds 1-2 and 1-3 before the locked confirmation fold. An eight-generation cache-hit run screened 160,000 genomes with 165,632 staged assay evaluations. Selected market gaps were `-1.066`, `-1.209`, `-1.480`, and `-2.019%p`; v4 lifts were `+9.270`, `+6.930`, `+7.401`, and `+6.812%p`. FAIL for the strict `+5.0%p` market gate; no production connection.

## Production decision

- `static/models/kra_model.joblib`은 `kra_dual_phase_v4_history_fresh_holdout_guard`를 유지한다.
- `pairwise.enabled=false`를 유지한다.
- H1~H11 intrinsic 후보 코드는 연구용이며 production scoring에 연결하지 않는다.
- 기존 v4 artifact의 `live_market_weight=1.0`은 변경하지 않는다.
- 앱의 KRA live market 경로는 그동안 `odds_snapshot_fresh`를 생성하는 호출부가 없어 사실상 비활성 상태였다. 변경안은 공식 KRA 주간 출발시각을 조회하고, 완전한 양수 배당판·결과 미존재·KST 출발 전 캡처를 모두 만족할 때만 해당 플래그를 생성한다. 일정 조회 실패나 조건 불충족 시 intrinsic v4로 fail closed 한다.
- 이 runtime gate 변경은 감사 전 커밋·푸시·배포하지 않는다.
- H13/H14 후보는 pure market 대비 `+5.0%p`를 충족하지 못해 runtime에 연결하지 않는다.
- H15 수집기는 모델 점수에 영향을 주지 않으며, 저장 실패가 예측을 중단하지 않는다.
- H16/H17의 odds-free sectional 및 세계 모델 후보는 `+5.0%p`에 미달해 production artifact와 runtime에 연결하지 않는다.
- H18은 v4 대비 절대 `+5.0%p`를 네 fold 모두 넘겼지만 시장 동률 게이트와 pristine-forward 게이트를 통과하지 못했으므로 production에 연결하지 않는다.
- H19~H21은 시장 초과 CI 및 pristine-forward 게이트를 통과하지 못했고 production에 연결하지 않는다.

## Evidence paths

- `runs/kra_max_winrate_hypotheses.md`
- `runs/kra_max_winrate_results.json`
- `runs/kra_max_winrate_h2_results.json`
- `runs/kra_max_winrate_h3_results.json`
- `runs/kra_max_winrate_h4_results.json`
- `runs/kra_max_winrate_h5_results.json`
- `runs/kra_max_winrate_h6_results.json`
- `runs/kra_max_winrate_h7_results.json`
- `runs/kra_max_winrate_h8_results.json`
- `runs/kra_max_winrate_h9_results.json`
- `runs/kra_max_winrate_h10_results.json`
- `runs/kra_max_winrate_h11_results.json`
- `runs/kra_max_winrate_h12_live_results.json`
- `runs/kra_max_winrate_h13_results.json`
- `runs/kra_max_winrate_h14_results.json`
- `runs/kra_max_winrate_h16_results.json`
- `runs/kra_max_winrate_h17_results.json`
- `runs/kra_drug_discovery_results.json`
- `runs/kra_drug_discovery_state.json`
- `runs/kra_drug_discovery_ledger.jsonl`
- `runs/kra_max_winrate_h20_results.json`
- `runs/kra_max_winrate_h21_results.json`
- `tools/kra_max_winrate_search.py`
- `tools/kra_diversified_search.py`
- `tools/kra_teacher_target_search.py`
- `tools/kra_meta_gate_search.py`
- `tools/kra_live_market_gate_report.py`
- `tools/kra_market_residual_search.py`
- `tools/kra_favorite_challenger_search.py`
- `tools/kra_sectional_search.py`
- `tools/kra_world_benchmark_search.py`
- `tools/kra_drug_discovery_search.py`
- `tools/kra_drug_discovery_loop.py`
- `tools/kra_market_pairwise_search.py`
- `kra_sectional_features.py`
- `kra_global_rankers.py`
- `kra_drug_discovery.py`
- `kra_drug_assay.py`
- `kra_drug_models.py`
- `kra_full_order_rankers.py`
- `kra_market_pairwise_challenger.py`
- `kra_market_context_gates.py`
- `kra_top1_screening.py`
- `kra_diversified_rankers.py`
- `kra_market_residual.py`
- `datastore.py`
- `engine.py`
- `app.py`
- `kra_dynamic_features.py`
- `kra_contextual_history.py`
- `kra_dynamic_ratings.py`
- `kra_pedigree_features.py`
- `kra_candidate_features.py`
- `kra_card_features.py`
- `runs/kra_hybrid_search_results.json`
- `runs/kra_hierarchical_search_results.json`
- `runs/kra_hierarchical_pareto_results.json`
- `kra_hierarchical_search.py`

## Audit questions

1. 같은 날 결과가 horse/jockey/trainer rating과 pedigree prior에 유입되지 않는가?
2. H1~H5 순차 탐색으로 fresh holdout이 반복 관찰되어 더 이상 pristine이라고 부를 수 없다는 한계를 문서화했는가?
3. `+5.0%p` 미달 후보가 어떠한 경로로도 production artifact에 연결되지 않는가?
4. 다음 승격 판정은 2026-07-12 이후 새로 쌓이는 미접촉 경주를 요구해야 하는가?
5. `parse_kra_race_starts`가 공식 주간표 행에서 날짜·개최지·경주번호·출발시각을 안전하게 매핑하고, 조회 실패 시 fail closed 하는가?
6. `kra_odds_snapshot_metadata`가 결과가 있는 행, 불완전 배당판, 출발 이후 캡처를 모두 차단하는가?
7. closing-odds retrospective ceiling과 timestamped pre-start forward evidence를 명확히 분리했는가?
8. H13/H14가 intrinsic v4가 아니라 pure market을 직접 기준으로 삼았는가?
9. H15 snapshot persistence가 저장 실패 시 예측을 깨지 않고, 동일 보드 중복 폭증을 막는가?

## Fresh verification

- Backend: `259` tests PASS after H24 implementation, including hierarchical stage accounting.
- Targeted KRA phase/live integration: PASS.
- Python compileall: PASS.
- Mobile TypeScript: PASS.
- Mobile security surface: PASS.
- `git diff --check`: PASS.
- Secret-pattern scan: PASS.
- H22 locked market-plus-five promotion validator: expected FAIL (all market gaps remain negative; pristine-forward gate not cleared).
- H23 hybrid-vs-wide comparison: expected FAIL (structured mixture coverage did not improve worst-fold market gap).
- H24 hierarchical Pareto-diversity search: expected FAIL for promotion (all four market gaps remain below the locked `+5.0%p` threshold; staged evaluations and report fields verified).

## Stop condition

Fable5 판정 전 커밋·푸시·배포 금지. 후보 승격은 새 미접촉 데이터와 `+5.0%p` 게이트 통과 전까지 금지한다.
