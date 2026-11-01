#!/bin/bash
# Entrypoint of the Renovate container (renovate.yml, docker-cmd-file). Renovate's image has no python,
# but renovate.json's postUpgradeTasks run `python3 ci/resolve_stack.py --apply` (stdlib only) and,
# for a vLLM bump, ci/check_constraints.py --update (needs python 3.12 + venv + pip, i.e. Ubuntu 24.04's
# python3). Install them as root, then run Renovate as its own user.
set -euo pipefail
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv python3-pip git >/dev/null
exec runuser -u ubuntu -- renovate
