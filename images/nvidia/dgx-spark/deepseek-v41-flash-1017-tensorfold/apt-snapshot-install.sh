#!/bin/bash
# Installs pinned Ubuntu 24.04 packages from a dated snapshot of the Ubuntu archive
# (https://snapshot.ubuntu.com), replacing every other apt source in the stage.
# Used by ./Dockerfile through a bind mount; it is not part of the final image.
set -euo pipefail

usage() {
  echo "Usage: $0 <snapshot YYYYMMDDTHHMMSSZ> <package=version>..." >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

snapshot="$1"
shift
if [[ ! "$snapshot" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  usage
  exit 2
fi

rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*
cat > /etc/apt/sources.list.d/ubuntu-snapshot.sources <<SOURCES
Types: deb
URIs: https://snapshot.ubuntu.com/ubuntu/${snapshot}
Suites: noble noble-updates noble-security
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
SOURCES

export DEBIAN_FRONTEND=noninteractive
apt-get -o Acquire::Check-Valid-Until=false update
apt-get install -y --no-install-recommends "$@"
rm -rf /var/lib/apt/lists/*
