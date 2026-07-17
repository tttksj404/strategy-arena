#!/usr/bin/env python3
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OracleDeployArtifactsTestCase(unittest.TestCase):
    def test_compose_enables_kcycle_live_and_persistent_search_outputs(self):
        compose = (ROOT / "deploy" / "oracle" / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("KCYCLE_ENABLED: \"1\"", compose)
        self.assertIn('RACELENS_FORCE_PRO: "${RACELENS_FORCE_PRO:-0}"', compose)
        self.assertIn('RACELENS_LIVE_DECISION_IP_PER_MIN_CAP: "${RACELENS_LIVE_DECISION_IP_PER_MIN_CAP:-60}"', compose)
        self.assertIn('RACELENS_FREE_DAILY_ANALYSIS_LIMIT: "${RACELENS_FREE_DAILY_ANALYSIS_LIMIT:-3}"', compose)
        self.assertIn('RACELENS_REWARDED_ADS_ENABLED: "${RACELENS_REWARDED_ADS_ENABLED:-0}"', compose)
        self.assertIn('RACELENS_ADMOB_REWARDED_AD_UNIT_ID: "${RACELENS_ADMOB_REWARDED_AD_UNIT_ID:-}"', compose)
        self.assertIn("KCYCLE_TRIFECTA_SNAPSHOT_PATH: /app/data/kcycle_trifecta_snapshots.jsonl", compose)
        self.assertIn("collector:", compose)
        self.assertIn("search-loop:", compose)
        self.assertIn('SEARCH_LOOP_INTERVAL_SEC: "${SEARCH_LOOP_INTERVAL_SEC:-30}"', compose)
        self.assertIn("sleep $${SEARCH_LOOP_INTERVAL_SEC:-30}", compose)
        self.assertIn("postgres:", compose)
        self.assertIn("mobile-web:", compose)
        self.assertIn("dockerfile: deploy/oracle/mobile.Dockerfile", compose)
        self.assertIn("RACELENS_UPSTREAM_API: http://app:8000", compose)
        self.assertIn("postgres_data:/var/lib/postgresql/data", compose)
        self.assertIn("DATABASE_URL: postgresql://strategy:${POSTGRES_PASSWORD}@postgres:5432/strategy_arena", compose)
        self.assertIn("strategy_data:/app/data", compose)
        self.assertIn("restart: unless-stopped", compose)

    def test_oracle_env_template_never_contains_real_secret(self):
        env_template = (ROOT / "deploy" / "oracle" / ".env.oracle.example").read_text(encoding="utf-8")

        self.assertIn("DATAGOKR_SERVICE_KEY=", env_template)
        self.assertIn("POSTGRES_PASSWORD=", env_template)
        self.assertNotIn("serviceKey=", env_template)
        self.assertNotIn("apikey", env_template.lower())

    def test_deploy_script_requires_explicit_oracle_host(self):
        deploy_script = (ROOT / "deploy" / "oracle" / "deploy.sh").read_text(encoding="utf-8")

        self.assertIn("ORACLE_HOST is required", deploy_script)
        self.assertIn("Oracle SSH is not reachable", deploy_script)
        self.assertIn("ORACLE_SSH_PORT", deploy_script)
        self.assertIn("SSH_OPTS", deploy_script)
        self.assertIn("-e \"$RSYNC_SSH\"", deploy_script)
        self.assertIn("resolve_datagokr_key", deploy_script)
        self.assertIn("read_remote_env_value", deploy_script)
        self.assertIn("remote_postgres_password", deploy_script)
        self.assertIn("remote_site_address", deploy_script)
        self.assertIn("remote_acme_email", deploy_script)
        self.assertIn("secrets.token_urlsafe", deploy_script)
        self.assertIn("POSTGRES_PASSWORD=$postgres_password", deploy_script)
        self.assertIn("ORACLE_SITE_ADDRESS=$site_address", deploy_script)
        self.assertIn("ACME_EMAIL=$acme_email", deploy_script)
        self.assertIn("$HOME/keirin/.env", deploy_script)
        self.assertIn("$HOME/kra/.env", deploy_script)
        self.assertIn("umask 077", deploy_script)
        self.assertIn("deploy/oracle/smoke.sh", deploy_script)
        self.assertIn("SEARCH_LOOP_INTERVAL_SEC=${SEARCH_LOOP_INTERVAL_SEC:-30}", deploy_script)
        self.assertIn("RACELENS_FORCE_PRO=${RACELENS_FORCE_PRO:-0}", deploy_script)
        self.assertIn("RACELENS_LIVE_DECISION_IP_PER_MIN_CAP=${RACELENS_LIVE_DECISION_IP_PER_MIN_CAP:-60}", deploy_script)
        self.assertIn("RACELENS_FREE_DAILY_ANALYSIS_LIMIT=${RACELENS_FREE_DAILY_ANALYSIS_LIMIT:-3}", deploy_script)
        self.assertIn("RACELENS_REWARDED_ADS_ENABLED=${RACELENS_REWARDED_ADS_ENABLED:-0}", deploy_script)
        self.assertIn("RACELENS_ADMOB_REWARDED_AD_UNIT_ID=${RACELENS_ADMOB_REWARDED_AD_UNIT_ID:-}", deploy_script)
        self.assertIn("remote_compose_cmd", deploy_script)
        self.assertIn("docker compose", deploy_script)
        self.assertIn("docker-compose", deploy_script)
        self.assertIn("--exclude \"deploy/oracle/.env.oracle\"", deploy_script)
        self.assertIn("detect_remote_oracle_path", deploy_script)
        self.assertIn("/home/ubuntu/strategy-arena", deploy_script)
        self.assertIn("Detected ORACLE_PATH=", deploy_script)
        self.assertIn("ORACLE_PATH is not set and no existing remote path was found", deploy_script)
        self.assertIn("smoke_attempt=1", deploy_script)
        self.assertIn("Smoke failed on attempt", deploy_script)
        self.assertIn("sleep 5", deploy_script)

    def test_prediction_search_loop_writes_to_persistent_data_volume(self):
        loop_script = (ROOT / "scripts" / "run_prediction_search_loop.sh").read_text(encoding="utf-8")

        self.assertIn("waiting_for_snapshots", loop_script)
        self.assertIn("--snapshots \"$snapshot_path\"", loop_script)
        self.assertIn("--out-json data/kcycle_trifecta_rule_search_results.json", loop_script)
        self.assertIn("--out-md data/kcycle_trifecta_rule_search_results.md", loop_script)
        self.assertIn("experiment_kcycle_late_market_pull.py", loop_script)
        self.assertIn("KCYCLE_LATE_MARKET_PULL_EXPERIMENT_ENABLED", loop_script)
        self.assertIn("experiment_kcycle_market_timing_policy.py", loop_script)
        self.assertIn("KCYCLE_MARKET_TIMING_EXPERIMENT_ENABLED", loop_script)
        self.assertIn("search_kcycle_fast_evolution_trifecta.py", loop_script)
        self.assertIn("KCYCLE_FAST_EVOLUTION_SEARCH_ENABLED", loop_script)
        self.assertIn("search_kcycle_global_breakthrough.py", loop_script)
        self.assertIn("KCYCLE_GLOBAL_BREAKTHROUGH_SEARCH_ENABLED", loop_script)
        self.assertIn("search_kcycle_drug_discovery_trifecta.py", loop_script)
        self.assertIn("${KCYCLE_DRUG_DISCOVERY_SEARCH_ENABLED:-0}", loop_script)

    def test_cloudshell_deploy_accepts_verified_reward_configuration(self):
        deploy_script = (ROOT / "deploy" / "oracle" / "cloudshell-deploy.sh").read_text(encoding="utf-8")

        self.assertIn('FREE_LIMIT="${RACELENS_FREE_DAILY_ANALYSIS_LIMIT:-3}"', deploy_script)
        self.assertIn('IP_CAP="${RACELENS_LIVE_DECISION_IP_PER_MIN_CAP:-60}"', deploy_script)
        self.assertIn('set_env RACELENS_FORCE_PRO "0" "$env_next"', deploy_script)
        self.assertIn('REWARDED_ADS_ENABLED="${RACELENS_REWARDED_ADS_ENABLED:-0}"', deploy_script)
        self.assertIn('REWARDED_AD_UNIT_ID="${RACELENS_ADMOB_REWARDED_AD_UNIT_ID:-}"', deploy_script)
        self.assertIn('set_env RACELENS_REWARDED_ADS_ENABLED "$rewarded_ads_enabled" "$env_next"', deploy_script)
        self.assertIn('set_env RACELENS_ADMOB_REWARDED_AD_UNIT_ID "$rewarded_ad_unit_id" "$env_next"', deploy_script)

    def test_oracle_smoke_enforces_production_contract(self):
        smoke_script = (ROOT / "deploy" / "oracle" / "smoke.sh").read_text(encoding="utf-8")

        self.assertIn('python3 - "$base_url"', smoke_script)
        self.assertIn("SMOKE_EXPECT_ENTITLEMENT", smoke_script)
        self.assertIn('expected_entitlement = os.environ.get("SMOKE_EXPECT_ENTITLEMENT", "free")', smoke_script)
        self.assertIn("healthz missing entitlement_mode", smoke_script)
        self.assertIn('health.get("entitlement_mode") != "production"', smoke_script)
        self.assertIn('health.get("ok") is not True', smoke_script)
        self.assertIn("mobile web root did not render RaceLens shell cleanly", smoke_script)
        self.assertIn("missing required store-review text", smoke_script)
        self.assertIn('app_session = parsed.get("app_session")', smoke_script)
        self.assertIn("app_session must be an object", smoke_script)
        self.assertIn('app-session entitlement must be {expected_entitlement}', smoke_script)
        self.assertIn('free_analysis_limit must be a positive integer', smoke_script)
        self.assertIn('free_analysis_remaining must be an integer from 0 to the limit', smoke_script)
        self.assertIn('initial_remaining = initial_session["free_analysis_remaining"]', smoke_script)
        self.assertIn('"date": "2026-07-03"', smoke_script)
        self.assertIn('settled live-decision decision must be settled', smoke_script)
        self.assertIn('2026-07-03 광명 1R response contains stale demo participant names', smoke_script)
        self.assertIn('if initial_remaining > 0:', smoke_script)
        self.assertIn('settled live-decision skipped (quota anchored)', smoke_script)
        self.assertIn('current_remaining = fetch_app_session("post-settled")["free_analysis_remaining"]', smoke_script)
        self.assertIn('if current_remaining > 0:', smoke_script)
        self.assertIn("timedelta(days=21)", smoke_script)
        self.assertIn("future no-race live-decision exceeded 5s fast-path budget", smoke_script)
        self.assertIn('future no-race live-decision decision must be hold', smoke_script)
        self.assertIn('future no-race live-decision status must not be blocked', smoke_script)
        self.assertIn('future no-race live-decision skipped (quota anchored)', smoke_script)
        self.assertIn("free quota changed after future no-race live-decision", smoke_script)
        self.assertIn('print("SMOKE2_DONE")', smoke_script)
        self.assertNotIn("Oracle smoke must expose Pro entitlement for the preview deployment", smoke_script)
        self.assertNotIn('free_analysis_remaining must be 3', smoke_script)
        self.assertNotIn('print("SMOKE_DONE")', smoke_script)

    def test_oracle_smoke_negative_fixture_rejects_missing_entitlement_mode(self):
        smoke_script = (ROOT / "deploy" / "oracle" / "smoke.sh").read_text(encoding="utf-8")

        self.assertIn("healthz missing entitlement_mode", smoke_script)
        self.assertIn('if "entitlement_mode" not in health:', smoke_script)
        self.assertIn('healthz entitlement_mode must be production', smoke_script)

    def test_korea_vps_deploy_script_is_provider_agnostic_and_fail_closed(self):
        deploy_script = (ROOT / "deploy" / "korea-vps" / "deploy.sh").read_text(encoding="utf-8")

        self.assertIn("VPS_HOST is required", deploy_script)
        self.assertIn("Korea VPS SSH is not reachable", deploy_script)
        self.assertIn("VPS_BOOTSTRAP_DOCKER", deploy_script)
        self.assertIn("bootstrap_ubuntu_docker.sh", deploy_script)
        self.assertIn("Docker Engine and the Docker Compose plugin are required", deploy_script)
        self.assertIn("resolve_datagokr_key", deploy_script)
        self.assertIn("SEARCH_LOOP_INTERVAL_SEC=${SEARCH_LOOP_INTERVAL_SEC:-0}", deploy_script)
        self.assertIn("read_remote_env_value", deploy_script)
        self.assertIn("remote_postgres_password", deploy_script)
        self.assertIn("$HOME/keirin/.env", deploy_script)
        self.assertIn("$HOME/kra/.env", deploy_script)
        self.assertIn("KCYCLE_ENABLED=1", deploy_script)
        self.assertIn("POSTGRES_PASSWORD=$postgres_password", deploy_script)
        self.assertIn("deploy/oracle/docker-compose.yml", deploy_script)
        self.assertIn("deploy/oracle/smoke.sh", deploy_script)

    def test_korea_vps_bootstrap_installs_docker_compose_on_ubuntu(self):
        bootstrap_script = (ROOT / "deploy" / "korea-vps" / "bootstrap_ubuntu_docker.sh").read_text(encoding="utf-8")

        self.assertIn("set -eu", bootstrap_script)
        self.assertIn("Ubuntu-compatible OS is required", bootstrap_script)
        self.assertIn("docker-ce", bootstrap_script)
        self.assertIn("docker-compose-plugin", bootstrap_script)
        self.assertIn("systemctl enable --now docker", bootstrap_script)


if __name__ == "__main__":
    unittest.main()
