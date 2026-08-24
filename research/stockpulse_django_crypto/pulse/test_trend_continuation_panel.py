import json
from pathlib import Path

from django.conf import settings
from django.test import TestCase


class TrendContinuationPanelContractTests(TestCase):
    def test_analysis_page_exposes_research_only_trend_panel(self):
        response = self.client.get("/analysis/")

        self.assertContains(response, 'id="bitgetTrendContinuationPanel"')
        self.assertContains(response, 'id="bitgetTrendContinuationList"')
        self.assertContains(response, 'id="bitgetTrendContinuationTitle"')
        self.assertContains(response, ">추세 지속 연구<")
        panel_markup = response.content.decode().split('id="bitgetTrendContinuationPanel"', 1)[1].split("</section>", 1)[0]
        self.assertIn("NO_TRADE", panel_markup)
        self.assertIn("검증 미완료", panel_markup)
        self.assertIn("실행 비활성", panel_markup)
        self.assertNotIn("<button", panel_markup)
        self.assertNotIn("bitgetLiveLongPanel", panel_markup)

    def test_client_source_keeps_research_panel_isolated_from_legacy_signal_language(self):
        script = Path(settings.BASE_DIR, "pulse", "static", "pulse", "analysis.js").read_text()
        trend_source = script.split("function trendContinuationDecisionLabel", 1)[1].split("async function loadBitgetLiveLongSignals", 1)[0]

        self.assertIn('"/api/bitget/trend-continuation/"', script)
        self.assertIn("NO_TRADE", trend_source)
        self.assertIn("검증 미완료", trend_source)
        self.assertIn("실행 비활성", trend_source)
        self.assertIn("시장 국면", trend_source)
        self.assertIn("특징", trend_source)
        self.assertIn("데이터 완전성", trend_source)
        self.assertIn("provenance", trend_source)
        self.assertIn("rejection_code", trend_source)
        self.assertIn("publicNoTrade", trend_source)
        self.assertIn("shouldRenderEmpiricalRate", trend_source)
        self.assertIn("renderTrendContinuationMeta", trend_source)
        self.assertIn("bitget-trend-meta-row", trend_source)
        self.assertIn("bitget-trend-status-chip", trend_source)
        self.assertIn("bitget-trend-time-token", trend_source)
        self.assertIn("bitget-trend-unavailable-copy", trend_source)
        self.assertIn("renderTrendContinuationTokenList", trend_source)
        for legacy_text in ("ENTER_LONG", "WATCH_LONG_SETUP", "CONTINUATION_OBSERVE", "관찰 후보", "상승 지속", "전진 검증 완료", "manual entry", "trade plan", "leverage", "order", "buy", "sell", "수동 진입", "주문", "매수", "매도"):
            self.assertNotIn(legacy_text, trend_source)

    def test_stylesheet_keeps_trend_panel_responsive_without_new_visual_system(self):
        stylesheet = Path(settings.BASE_DIR, "pulse", "static", "pulse", "app.css").read_text()

        self.assertIn(".bitget-trend-continuation-panel", stylesheet)
        self.assertIn(".bitget-trend-evidence-grid", stylesheet)
        self.assertIn("repeat(auto-fit, minmax(150px, 1fr))", stylesheet)
        self.assertIn(".bitget-trend-token-list", stylesheet)
        self.assertIn(".bitget-trend-token", stylesheet)
        self.assertIn(".bitget-trend-meta-row", stylesheet)
        self.assertIn(".bitget-trend-status-chip", stylesheet)
        self.assertIn(".bitget-trend-quote-row", stylesheet)
        self.assertIn(".bitget-trend-time-token", stylesheet)
        self.assertIn("white-space: nowrap;", stylesheet)
        self.assertIn(".bitget-trend-unavailable-copy", stylesheet)
        self.assertIn("overflow-wrap: anywhere;", stylesheet)
        self.assertIn("@media (max-width: 420px)", stylesheet)
        self.assertIn(".bitget-trend-evidence-grid {\n    grid-template-columns: minmax(0, 1fr);", stylesheet)
        self.assertIn("var(--font-size-number-display)", stylesheet)

    def test_package_js_check_covers_trend_panel_scripts(self):
        package_json = json.loads(Path(settings.BASE_DIR, "package.json").read_text())
        check_js = package_json["scripts"]["check:js"]

        self.assertIn("scripts/verify_trend_continuation_panel.mjs", check_js)
        self.assertIn("scripts/run_trend_continuation_panel_smoke.mjs", check_js)
        self.assertIn("scripts/verify_trend_continuation_server_lifecycle.mjs", check_js)
