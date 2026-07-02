#!/usr/bin/env python3
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OracleDeployArtifactsTestCase(unittest.TestCase):
    def test_compose_enables_kcycle_live_and_persistent_search_outputs(self):
        compose = (ROOT / "deploy" / "oracle" / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("KCYCLE_ENABLED: \"1\"", compose)
        self.assertIn("KCYCLE_TRIFECTA_SNAPSHOT_PATH: /app/data/kcycle_trifecta_snapshots.jsonl", compose)
        self.assertIn("collector:", compose)
        self.assertIn("search-loop:", compose)
        self.assertIn("strategy_data:/app/data", compose)
        self.assertIn("restart: unless-stopped", compose)

    def test_oracle_env_template_never_contains_real_secret(self):
        env_template = (ROOT / "deploy" / "oracle" / ".env.oracle.example").read_text(encoding="utf-8")

        self.assertIn("DATAGOKR_SERVICE_KEY=", env_template)
        self.assertNotIn("serviceKey=", env_template)
        self.assertNotIn("apikey", env_template.lower())

    def test_deploy_script_requires_explicit_oracle_host(self):
        deploy_script = (ROOT / "deploy" / "oracle" / "deploy.sh").read_text(encoding="utf-8")

        self.assertIn("ORACLE_HOST is required", deploy_script)
        self.assertIn("Oracle SSH is not reachable", deploy_script)
        self.assertIn("resolve_datagokr_key", deploy_script)
        self.assertIn("$HOME/keirin/.env", deploy_script)
        self.assertIn("$HOME/kra/.env", deploy_script)
        self.assertIn("umask 077", deploy_script)
        self.assertIn("deploy/oracle/smoke.sh", deploy_script)
        self.assertIn("docker compose -f deploy/oracle/docker-compose.yml", deploy_script)
        self.assertIn("--exclude \"deploy/oracle/.env.oracle\"", deploy_script)

    def test_prediction_search_loop_writes_to_persistent_data_volume(self):
        loop_script = (ROOT / "scripts" / "run_prediction_search_loop.sh").read_text(encoding="utf-8")

        self.assertIn("waiting_for_snapshots", loop_script)
        self.assertIn("--snapshots \"$snapshot_path\"", loop_script)
        self.assertIn("--out-json data/kcycle_trifecta_rule_search_results.json", loop_script)
        self.assertIn("--out-md data/kcycle_trifecta_rule_search_results.md", loop_script)

    def test_oracle_smoke_enforces_live_decision_contract(self):
        smoke_script = (ROOT / "deploy" / "oracle" / "smoke.sh").read_text(encoding="utf-8")

        self.assertIn("live-decision missing fields", smoke_script)
        self.assertIn("market_odds must be a list", smoke_script)
        self.assertIn("market_risk.level=odds_live", smoke_script)
        self.assertIn("live_market_blocked risk", smoke_script)


if __name__ == "__main__":
    unittest.main()
