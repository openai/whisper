# Security Policy

We take the security of Whisper seriously. If you believe you have found a security vulnerability in this repository, please report it to us using the instructions below.

## Supported Versions

Security updates are actively provided for the latest release on the `main` branch. 

| Version | Supported |
| ------- | --------- |
| Latest (`main` branch) | :white_check_mark: |
| < 2025.x | :x: |

Please ensure you are using the latest version of Whisper to receive all security patches.

## Reporting a Vulnerability

**Please do not report security vulnerabilities via public GitHub issues.**

If you discover a security vulnerability in this repository, please report it through one of the following channels:
* **Disclosure Portal:** [OpenAI Coordinated Vulnerability Disclosure](https://openai.com/security/disclosure/)
* **Email:** disclosure@openai.com

When submitting a report, please include:
1. A detailed description of the vulnerability.
2. Step-by-step instructions or a proof-of-concept (PoC) to reproduce the issue.
3. The impact of the vulnerability and potential mitigation strategies.

We will acknowledge your submission and coordinate a fix and disclosure timeline.

## Security Guidelines for Users
* **Model Loading:** Whisper uses PyTorch models. Loading models from untrusted sources can lead to arbitrary code execution (unpickling exploits). Only load official OpenAI models or models from trusted entities.
* **Audio Input Validation:** Ensure any user-provided audio files are processed through a validated media framework (such as `ffmpeg` with up-to-date security patches) before passing to Whisper.
* **Dependencies:** Keep dependencies (especially `torch`, `numpy`, and `ffmpeg`) updated to their latest secure version
