# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Timber, please report it privately.

- Email: **kossi@electricsheep.africa**
- Subject: **[Timber Security] <short description>**

Please include:

1. A clear description of the issue
2. Steps to reproduce
3. Affected version(s)
4. Potential impact
5. Any suggested mitigation

## Response Process

- We will acknowledge receipt within **72 hours**.
- We will investigate and triage severity.
- We will work on a fix and coordinate responsible disclosure.
- We will publish a security advisory once a patch is available.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes |
| < 0.1.0 | No |

## Scope

Security reports are especially valuable for:

- Unsafe model artifact parsing
- Remote code execution paths in CLI/server
- Memory safety problems in generated C runtime
- Authentication/authorization issues (if introduced in future server modes)
- Supply-chain risks in build/release workflows
