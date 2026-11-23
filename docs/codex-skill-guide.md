# Codex 스킬·지침 가이드 — strategy-arena

> 생성일 2026-11-23. Codex 작업 폴더 `~/Desktop/2학기/strategy-arena`에서 이 레포를 다룰 때 적용되는 지침과 자주 쓰는 스킬이다.
> 전체 목록(사용자 스킬 66개, 플러그인 스킬, 모든 작업 폴더 맵)은 `tttksj404/codex-global-skills`의 `docs/codex-skill-usage-guide.md`에 있다.

## 적용 지침

| 지침 | 언제 | 내용 |
|---|---|---|
| 전역 `~/.codex/AGENTS.md` | 모든 작업 | 한국어 답변, 직접 수행·검증, 결과/검증/남은 리스크 보고, 복잡한 작업은 `codex-fablize` 자동 승격 |
| 전역 "라우팅과 자동 승격" | 전략·백테스트·리스크 코드 | 금융은 Fable5 감사 조건, 확정 데이터 뒷북 판단 금지 |

## 이 레포에서 주로 쓰는 스킬

"언제 쓰나"는 각 `SKILL.md`의 description 원문이다.

| 스킬 | 언제 쓰나 (SKILL.md description) |
|---|---|
| `codex-fablize` | Use when the user asks for fablize, codex-fablize, gpt55-fablize (legacy name), Fable-style, Fable5-style, or to make Codex work like Fable — on any Codex model, not only GPT-5.5. Applies only transferable workflow: inspect first, finish multi-step work, checkpoint evidence, investigate systematically, observe rendered artifacts, verify before done, and separate implementation from audit roles. Does not clone Clau... |
| `binance-api-skill` | Implements/debugs Binance Spot REST /api/* calls: /api/v3 market data, signed account/orders, keys, HMAC/RSA/Ed25519 signing, recvWindow, filters, rate limits, errors, Testnet. |
| `jupyter-notebook` | Use when the user asks to create, scaffold, or edit Jupyter notebooks (`.ipynb`) for experiments, explorations, or tutorials; prefer the bundled templates and run the helper script `new_notebook.py` to generate a clean starting notebook. |
| `token-quality-engine` | Reduce context usage without lowering the model or reasoning setting through hash-verified log capsules and exact-line context virtualization. Use for large repetitive logs and large structured code-navigation, code-review, specification, decision-history, debugging, or dependency-planning contexts. Bypass dense tables, security or financial evidence, unstructured user data, short inputs, and any task requiring th... |
| `devil` | Devil's Advocate code/prose judge. Use when the user wants harsh, evidence-based scoring (10.0 scale) of code or writing — refuses sycophancy, demands cited deductions for every score below 1.5/sport. Activates Comrade-Judge persona that auto-detects code mode (7 sports × 1.5pt) or prose mode (5 sports × 2.0pt). Triggers: /devil, "엄격하게 평가", "독한 리뷰", "devil's advocate", "harsh review". |
| `arena` | Run Arena Mode — spawn N agents on the same task, rank their solutions, and return the best. Use for ambiguous bugs, design trade-offs, refactors, or optimizations with multiple valid approaches. |
| `simon-review` | Use when the user wants code, ideas, or marketing plans reviewed with sharp VC-perspective judgment. Applies Kim Seo-jun (Hashed GP) 9 decision-making principles to find structural flaws, blind spots, and missing questions. Goal is accurate critique, not style mimicry. |
