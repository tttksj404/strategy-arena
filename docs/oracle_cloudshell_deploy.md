# Oracle Cloud Shell Deploy Path

This is the permanent deploy path for RaceLens on Oracle.

## Why This Path Exists

- The production instance is `ubuntu@168.107.2.218`.
- Windows direct SSH to the instance and OCI Bastion host can be blocked by the local network.
- OCI Run Command is not reliable for this instance because commands can remain `ACCEPTED` and expire when the instance agent does not poll.
- OCI Cloud Shell can reach the instance on SSH.

Use Cloud Shell as the stable deploy jump host.

GitHub Actions is the second permanent path. It keeps a separate deploy key in
repository secrets, so deployment still works even if Cloud Shell home state is
reset.

## One-Time State Already Installed

Cloud Shell has a persistent RSA key:

```bash
~/.ssh/racelens_oracle_rsa
```

The matching public key is installed in:

```bash
ubuntu@168.107.2.218:/home/ubuntu/.ssh/authorized_keys
```

The key is RSA because Cloud Shell FIPS mode rejects ed25519 key generation.

GitHub Actions also has a dedicated RSA deploy key. Its public key comment is:

```text
racelens-github-actions-deploy
```

The private key is stored only as the GitHub repository secret:

```text
ORACLE_SSH_PRIVATE_KEY
```

## Daily Deploy

Open OCI Cloud Shell in the Chuncheon region and run:

```bash
~/bin/racelens-deploy --deploy
```

Health-only check:

```bash
~/bin/racelens-deploy --check
```

The script:

- clones `main` from `https://github.com/tttksj404/strategy-arena.git`
- preserves the current server `deploy/oracle/.env.oracle`
- pins `RACELENS_LIVE_DECISION_IP_PER_MIN_CAP=60`
- pins `RACELENS_FREE_DAILY_ANALYSIS_LIMIT=3`
- builds before swapping the live source directory
- moves the previous deployment to `/home/ubuntu/strategy-arena.backup.<timestamp>`
- restarts Docker Compose project `oracle`
- runs local health and `deploy/oracle/smoke.sh`

## GitHub Actions Deploy

Use **Actions -> Oracle Deploy -> Run workflow**.

Defaults are production-safe:

- `branch`: `main`
- `free_limit`: `3`
- `ip_cap`: `60`
- `base_url`: `https://168-107-2-218.sslip.io`

The workflow uses the same script as Cloud Shell:

```text
.github/workflows/oracle-deploy.yml
deploy/oracle/cloudshell-deploy.sh
```

It checks out the requested ref, writes `ORACLE_SSH_PRIVATE_KEY` to an ephemeral
runner file, deploys over SSH, runs `/healthz`, `/legal/support`, and
`deploy/oracle/smoke.sh`, then deletes the runner key file.

## Reinstall Script In Cloud Shell

If `~/bin/racelens-deploy` disappears but the key remains:

```bash
mkdir -p ~/bin
curl -fsSL https://raw.githubusercontent.com/tttksj404/strategy-arena/main/deploy/oracle/cloudshell-deploy.sh -o ~/bin/racelens-deploy
chmod 700 ~/bin/racelens-deploy
~/bin/racelens-deploy --check
```

## SSH Details

Manual SSH from Cloud Shell:

```bash
ssh -i ~/.ssh/racelens_oracle_rsa \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=12 \
  -o PubkeyAcceptedKeyTypes=+ssh-rsa,rsa-sha2-256,rsa-sha2-512 \
  ubuntu@168.107.2.218
```

Do not rely on Windows direct SSH or Run Command as the primary deployment path.
