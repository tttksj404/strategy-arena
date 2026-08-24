from pathlib import Path

from django.conf import settings
from django.test import TestCase


class BitgetLiveSignalPanelTemplateTests(TestCase):
    def test_analysis_page_exposes_the_live_manual_long_signal_panel(self):
        response = self.client.get("/analysis/")

        self.assertContains(response, 'id="bitgetLiveLongPanel"')
        self.assertContains(response, 'id="bitgetLiveSignalList"')
        self.assertContains(response, "60초 갱신")
        panel_markup = response.content.decode().split('id="bitgetLiveLongPanel"', 1)[1].split("</section>", 1)[0]
        self.assertNotIn("<button", panel_markup)

    def test_client_copy_keeps_the_panel_manual_and_non_actionable(self):
        script = Path(settings.BASE_DIR, "pulse", "static", "pulse", "analysis.js").read_text()

        self.assertIn("지금 수동 진입 조건 충족", script)
        self.assertIn("take_profit_1_price", script)
        self.assertIn("take_profit_2_price", script)
        self.assertIn("호가 스프레드 과다", script)
        self.assertIn("폐봉 신호 경과", script)
        self.assertIn("estimated_all_in_margin_loss_at_stop_pct", script)
        self.assertIn('profile === "triple_leveraged_15m_5m_1m"', script)
        self.assertIn("검증 전 관찰 조건 충족", script)
        self.assertIn("watch_candidates", script)
        self.assertIn("현재 공개 호가와 비용 기준을 함께 통과한 롱 설정이 없습니다.", script)
        self.assertIn("로그인 또는 데모 세션에서 읽기 전용 롱 신호를 확인할 수 있습니다.", script)
        self.assertIn("주문이나 계정 정보는 조회하지 않습니다.", script)
        self.assertNotIn("ENTER_LONG 신호", script)

    def test_panel_uses_shared_number_tokens_and_has_a_narrow_mobile_layout(self):
        stylesheet = Path(settings.BASE_DIR, "pulse", "static", "pulse", "app.css").read_text()

        self.assertIn("--font-number-display:", stylesheet)
        self.assertIn("font-size: var(--font-size-number-display);", stylesheet)
        self.assertIn("font-weight: var(--font-weight-number-display);", stylesheet)
        self.assertIn("@media (max-width: 420px)", stylesheet)
        self.assertIn(".bitget-live-levels {\n    grid-template-columns: minmax(0, 1fr);", stylesheet)
