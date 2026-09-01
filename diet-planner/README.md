# AI Diet Planner (school project)

A fully portable, zero-installation web app that generates a personalised 7-day
meal plan. It runs by **double-clicking `index.html`** — straight from a USB
drive over the `file://` protocol. No server, no build step, no admin rights,
no Node/Python. Plain HTML + CSS + JavaScript in a single file.

## Files

| File | Purpose |
|---|---|
| `index.html` | The whole app (inline CSS + JS) |
| `test.html` | Minimal connectivity test for the Gemini API from `file://` |
| `README.md` | This file |

## Setup (2 steps)

1. **Get a free Gemini API key** at <https://aistudio.google.com/apikey>
   (Google account required; free tier is enough — the app uses a Flash model).
2. **Open `index.html` in any text editor** (Notepad works) and paste the key
   into the clearly marked constant near the top of the `<script>` block:

   ```js
   const API_KEY = "PASTE_YOUR_KEY_HERE";
   ```

That's it. Double-click `index.html` and generate plans. Without a key the app
still works — it just stays in offline mode (see below).

### Sanity check first (optional but recommended)

Double-click `test.html`. It makes one `fetch()` call to the Gemini REST API:

* **"SUCCESS"** — everything works end to end.
* **"CORS OK (HTTP 400 …)"** — the API is reachable from `file://`; you just
  haven't pasted a valid key into `test.html` yet.
* **"FETCH FAILED" / "TIMED OUT"** — the network (e.g. school wifi) blocks the
  API. The main app will automatically run in offline mode there.

## Why this works from `file://` (no CORS problem)

Browsers send `Origin: null` for pages opened from disk. Google's Gemini
endpoint (`generativelanguage.googleapis.com`) explicitly answers with
`Access-Control-Allow-Origin: null` and allows the `Content-Type` and
`x-goog-api-key` headers, so cross-origin `fetch()` from a `file://` page is
permitted by the API. No local server or proxy is needed.

## How the offline fallback works

The app **always** computes the calorie target locally, then tries to get the
meal plan from Gemini. It drops to the built-in rule-based planner — and shows
an **"Offline mode"** badge — whenever any of these happen:

* no API key is pasted in,
* the request fails (wifi blocked, DNS error, CORS proxy at school, …),
* the request takes longer than **15 seconds** (AbortController timeout),
* the API returns **HTTP 429** (rate limit) and one retry after 3 seconds
  also fails,
* the AI's reply isn't valid JSON in the expected shape.

There is also a **"Force offline mode"** checkbox for deliberately demoing the
fallback. Two extra robustness features on the AI path:

* **Model discovery** — model names change often; if the configured model
  returns HTTP 404, the app calls the API's `ListModels` endpoint and picks a
  currently available Flash model automatically.
* **Strict validation** — the AI is asked for pure JSON
  (`responseMimeType: "application/json"`), and the reply is parsed and
  shape-checked before rendering; anything malformed triggers the fallback.

### Offline planner rules

1. **BMR** via Mifflin-St Jeor:
   * men: `10·kg + 6.25·cm − 5·age + 5`
   * women: `10·kg + 6.25·cm − 5·age − 161`
2. **TDEE** = BMR × activity multiplier
   (sedentary 1.2, light 1.375, moderate 1.55, very active 1.725, extra active 1.9).
3. **Goal adjustment**: lose −500 kcal, maintain 0, gain +400 kcal.
4. **Safety clamp**: the daily target never goes below **1200 kcal**; if the
   raw number is lower, it is raised and a warning is shown.
5. The target is split 25% breakfast / 30% lunch / 30% dinner / 15% snack, and
   meals are chosen from a built-in database of 32 items (each with calories,
   protein, carbs, fat, allergen tags, cuisine tag and ingredient list),
   filtered by diet type (vegan ⊂ vegetarian ⊂ non-veg) and allergies,
   preferring the chosen cuisine, rotating choices so all 7 days differ, and
   scaling portion sizes (×0.65–×1.85) to hit each meal's calorie budget.
6. The grocery list is the union of ingredients across all 28 chosen meals.

The same clamp and calorie math also feed the AI prompt, so both modes aim at
the same target.

## Architecture (for the project write-up)

```
index.html
├── UI layer          form inputs → readInputs(); results rendered as an
│                     HTML table + grocery list; print stylesheet hides the
│                     form so window.print() produces a clean PDF
├── Calorie engine    calorieTarget(): Mifflin-St Jeor + activity + goal,
│                     clamped to ≥1200 kcal (shared by both modes)
├── AI mode           buildPrompt() → geminiGenerate(): plain fetch() to the
│                     Gemini REST API with a 15 s timeout, one 3 s-delayed
│                     retry on HTTP 429, and ListModels-based model discovery
│                     on HTTP 404 → parseAiPlan() validates the JSON
└── Offline mode      offlinePlan(): deterministic rule-based planner over a
                      32-item meal database (diet/allergy/cuisine filters,
                      day rotation, portion scaling, grocery aggregation)
```

Design decisions worth mentioning:

* **Single file, no dependencies** — survives locked-down PCs and USB drives;
  nothing to install and no network needed except the optional AI call.
* **Graceful degradation** — the AI is an enhancement, not a requirement; every
  failure path lands in a working offline planner with a visible mode badge.
* **Client-side day totals** — per-day calorie/macro totals are always summed
  in the browser from the individual meals, so the table is internally
  consistent even if the AI's own arithmetic is off.
* **Escaped rendering** — all AI-supplied text is HTML-escaped before being
  inserted into the page.

## Security & privacy notes

* The API key sits in client-side source. That is acceptable for a personal
  school demo, but never publish the file (or the USB stick) with a real key
  inside, and restrict/rotate the key in Google AI Studio if it leaks.
* Everything the user types stays in the browser except the plan request sent
  to Google when AI mode is active.

## Disclaimer

**Educational project — not medical advice.** The plans are generated for a
school assignment and must not be used to treat any medical condition.
