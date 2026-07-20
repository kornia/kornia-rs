#!/usr/bin/env python3
"""Fail if any kornia-* pin in [workspace.dependencies] disagrees with
workspace.package.version.

The workspace crates inherit their own version via `version.workspace =
true`, but the inter-crate dependency pins in [workspace.dependencies]
are plain strings Cargo cannot inherit — and locally the `path`
component wins, so a stale pin builds fine and only ships a wrong
requirement at publish time (this bit the v0.1.15-rc.5 cut). Run from
pre-commit / CI so a release bump can never touch one without the
other.
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "Cargo.toml"
text = root.read_text()

m = re.search(r'^\[workspace\.package\]$.*?^version = "([^"]+)"', text, re.M | re.S)
if not m:
    sys.exit("workspace.package.version not found in Cargo.toml")
ws_version = m.group(1)

bad = []
for line in text.splitlines():
    pin = re.match(r'^(kornia[a-z0-9-]*)\s*=.*path = "crates/.*version = "([^"]+)"', line)
    if pin and pin.group(2) != ws_version:
        bad.append(f"  {pin.group(1)}: {pin.group(2)} (workspace is {ws_version})")

if bad:
    print("workspace-dependency version pins out of sync with workspace.package.version:")
    print("\n".join(bad))
    print(f'fix: sed -i \'s/version = "<stale>"/version = "{ws_version}"/g\' Cargo.toml')
    sys.exit(1)
print(f"version pins OK ({ws_version})")
