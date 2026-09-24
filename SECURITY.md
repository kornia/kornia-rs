# Security Policy

## Supported Versions

kornia-rs is pre-1.0. Security fixes are made on the `main` branch and shipped
in the next release. Only the **latest published release** (crates.io `kornia*`
crates and the `kornia-rs` PyPI package) receives security fixes; please upgrade
before reporting issues against older versions.

| Version        | Supported |
| -------------- | --------- |
| Latest release | Yes       |
| Older releases | No        |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Report them privately via GitHub Private Vulnerability Reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Fill in the advisory form.

If Private Vulnerability Reporting is unavailable, contact a project maintainer
privately (for example via the contact details on their GitHub profile) and ask
for a private channel. Do not include vulnerability details in public channels.

Please include as much of the following as you can:

- Affected crate(s) / package and version(s)
- A description of the issue and its impact
- Steps to reproduce or a proof of concept
- Any suggested fix or mitigation

## What to Expect

- **Acknowledgement:** within 7 days of your report.
- **Initial assessment:** within 14 days, including whether we accept the report.
- **Fix and disclosure:** we aim to release a fix within 90 days, coordinating
  the disclosure date with you and publishing a GitHub Security Advisory (and a
  RustSec advisory where applicable). We are happy to credit reporters who wish
  to be named.

This is a volunteer-maintained project, so timelines are best effort.
