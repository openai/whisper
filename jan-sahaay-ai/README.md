# Jan Sahaay AI 🇮🇳

**India's first voice-first app for welfare schemes.**

Jan Sahaay AI helps every Indian discover the government schemes they are
eligible for — by simply **speaking** about themselves in Hindi, English or
10 other Indian languages, or by filling a short form. It ships with a
database of **1,071 schemes** covering the Central Government, **Jharkhand**,
and **all 28 states + 8 union territories**.

## ✨ Features

- 🎙 **Voice-first**: tap the mic and say, in Hindi or English,
  *"मैं झारखंड की 35 साल की महिला किसान हूं, आय 2 लाख"* — the app extracts
  your age, gender, state, occupation, income, category and special
  circumstances (widow, disability, BPL, pregnancy) and instantly shows
  matching schemes. 12 Indian voice languages are selectable.
- 🌐 **Fully bilingual UI** — one tap switches the entire app, including
  every scheme name and benefit, between English and हिन्दी.
- 🗂 **1,000+ schemes** across categories: agriculture, health, education,
  housing, pensions, women & child, skills & jobs, business loans,
  insurance, food security, workers' welfare and social welfare.
- 🎯 **Eligibility engine** — filters by age, gender, state/UT, rural/urban,
  income, social category, occupation and special flags; results are ranked
  with your state's schemes first and can be filtered by category.
- 🔗 Every result links to **myscheme.gov.in** for verification and
  application.
- 🖥 **Futuristic glass UI** — animated aurora background, glassmorphism
  panels, tricolor-neon accents, fully responsive.
- 📦 **Single .exe** — no installation, no dependencies; runs completely
  offline except the voice recognition (which uses the browser's built-in
  speech service).

## 📥 Getting the Windows .exe

The exe is built automatically by GitHub Actions (`Build Jan Sahaay AI
Windows EXE` workflow) on every push touching `jan-sahaay-ai/`:

1. Open the repo's **Actions** tab → *Build Jan Sahaay AI Windows EXE* →
   latest green run.
2. Download the **JanSahaayAI-windows-exe** artifact.
3. Unzip and double-click `JanSahaayAI.exe`. It starts a local server and
   opens the app in your default browser (use Chrome/Edge for voice input).
   Closing the tab exits the app automatically.

You can also trigger the workflow manually from the Actions tab
(*Run workflow*), or build locally on any Windows machine:

```bat
pip install pyinstaller
cd jan-sahaay-ai
python generator/generate_schemes.py
pyinstaller --onefile --noconsole --name JanSahaayAI --add-data "app;app" launcher.py
:: → dist\JanSahaayAI.exe
```

## 🚀 Running from source (any OS)

```bash
cd jan-sahaay-ai
python3 launcher.py          # serves the app and opens your browser
```

or just open `app/index.html` directly in Chrome/Edge.

## 🗄 Regenerating the scheme database

```bash
python3 generator/generate_schemes.py   # rewrites app/schemes-data.js
```

The database combines:

- **Curated Central Government schemes** (PM-KISAN, Ayushman Bharat,
  Mudra, PMAY, scholarships, pensions, insurance, …)
- **Curated flagship state schemes** — with special depth for Jharkhand
  (Maiya Samman, Abua Awas, Guruji Credit Card, Sarvajan Pension, …) plus
  major schemes of 20+ other states/UTs
- **State-wise implementations of Centrally Sponsored Schemes** (NSAP
  pensions, PMAY-G/U, MGNREGA, NFSA, scholarships, PM-JAY, BOCW welfare,
  skill missions, KCC, PMFBY, …) for **every** state and UT, since these
  are applied for through state departments.

> **Disclaimer**: scheme details are indicative and change over time.
> Users must verify eligibility and apply only through official portals —
> [myscheme.gov.in](https://www.myscheme.gov.in) or the relevant state
> government website.

## 🧱 Architecture

```
jan-sahaay-ai/
├── app/                    # the web app (offline, zero dependencies)
│   ├── index.html          # UI skeleton
│   ├── style.css           # futuristic glassmorphism theme
│   ├── app.js              # i18n, voice, NLU parsing, matching engine
│   └── schemes-data.js     # generated database (1,071 schemes)
├── generator/
│   └── generate_schemes.py # database generator
├── launcher.py             # local server + browser launcher (→ .exe)
└── README.md
```

No frameworks, no build step, no network calls (except optional voice
recognition and the MyScheme verification links) — so the packaged exe is
small, fast and private: **no user data ever leaves the device**.
