/* ============ Jan Sahaay AI — application logic ============ */
"use strict";

/* ---------------- i18n ---------------- */
const I18N = {
  en: {
    tagline: "India's first voice-first app for welfare schemes",
    heroA: "Speak.", heroB: "Discover.", heroC: "Claim your rights.",
    heroSub: "Tell us about yourself — by voice or by form — and instantly discover every government scheme you are eligible for, across India.",
    stSchemes: "Schemes", stStates: "States & UTs", stLangs: "Voice languages", stFree: "Free forever",
    voiceTitle: "Voice Assistant",
    voiceIdle: "Tap the mic and speak — e.g. “I am a 35 year old woman farmer from Jharkhand, income 2 lakh”",
    voiceListening: "Listening… speak now",
    voiceUnsupported: "Voice input needs Google Chrome or Microsoft Edge (and an internet connection). You can still use the form below.",
    formTitle: "Your Profile", reset: "Reset",
    fAge: "Age", fGender: "Gender", fState: "State / UT", fArea: "Area",
    fCat: "Social category", fOcc: "Occupation", fIncome: "Annual family income (₹)",
    fBpl: "BPL / Antyodaya ration card", fDis: "Person with disability",
    fWid: "Widow", fPreg: "Pregnant / new mother",
    oSelect: "— Select —", oFemale: "Female", oMale: "Male", oOther: "Other / Transgender",
    oRural: "Rural (village)", oUrban: "Urban (town/city)",
    oGen: "General", oObc: "OBC", oSc: "SC", oSt: "ST", oEws: "EWS", oMin: "Minority",
    oStudent: "Student", oFarmer: "Farmer", oWorker: "Daily-wage / construction worker",
    oSalaried: "Salaried employee", oSelf: "Business / self-employed",
    oVendor: "Street vendor", oArtisan: "Artisan / craftsperson",
    oFisher: "Fisherman / fisherwoman", oHome: "Homemaker", oUnemp: "Unemployed", oRetired: "Retired",
    lakh: "lakh", cta: "Find My Schemes",
    resultsFound: n => `You may be eligible for <em>${n}</em> schemes`,
    resultsNone: "No matching schemes found",
    noResultsMsg: "Try filling fewer fields, or check the income entered. Every field left blank is treated as “any”.",
    allTags: "All",
    verify: "View on MyScheme →",
    ageChip: a => `Age: ${a}`, incomeChip: i => `Income: ₹${Number(i).toLocaleString("en-IN")}`,
    disclaimer: "Scheme details are indicative and may change. Always verify eligibility and apply through official portals — myscheme.gov.in or your state government website.",
    footer: "Empowering every Indian with knowledge of their entitlements",
    levels: { C: "Central", S: "State", CSS: "Centrally Sponsored" },
    tags: { agri: "Agriculture", health: "Health", edu: "Education", housing: "Housing & Utilities",
      pension: "Pension", wcd: "Women & Child", skill: "Skills & Jobs", biz: "Business & Loans",
      ins: "Insurance", fin: "Savings & Banking", food: "Food Security", worker: "Workers",
      social: "Social Welfare" },
    chips: { age: (a, b) => a != null && b != null ? `Age ${a}–${b}` : (a != null ? `Age ${a}+` : `Up to age ${b}`),
      inc: v => `Income ≤ ₹${Number(v).toLocaleString("en-IN")}`,
      F: "Women", M: "Men", O: "Transgender", R: "Rural", U: "Urban",
      bpl: "BPL", dis: "Disability", wid: "Widow", preg: "Maternity",
      sow: "SC/ST or Women", allIndia: "All India",
      cat: { sc: "SC", st: "ST", obc: "OBC", gen: "General", ews: "EWS", minority: "Minority" },
      occ: { student: "Students", farmer: "Farmers", worker: "Workers", salaried: "Salaried",
        self_employed: "Self-employed", unemployed: "Unemployed", homemaker: "Homemakers",
        fisher: "Fishers", artisan: "Artisans", street_vendor: "Street vendors", retired: "Retired" } },
  },
  hi: {
    tagline: "कल्याणकारी योजनाओं के लिए भारत का पहला वॉइस-फर्स्ट ऐप",
    heroA: "बोलिए।", heroB: "जानिए।", heroC: "अपना हक़ पाइए।",
    heroSub: "अपने बारे में बताइए — आवाज़ से या फ़ॉर्म से — और तुरंत जानिए कि पूरे भारत में आप किन-किन सरकारी योजनाओं के पात्र हैं।",
    stSchemes: "योजनाएं", stStates: "राज्य व केंद्रशासित प्रदेश", stLangs: "वॉइस भाषाएं", stFree: "हमेशा निःशुल्क",
    voiceTitle: "वॉइस असिस्टेंट",
    voiceIdle: "माइक दबाकर बोलिए — जैसे “मैं झारखंड की 35 साल की महिला किसान हूं, आय 2 लाख”",
    voiceListening: "सुन रहे हैं… अब बोलिए",
    voiceUnsupported: "वॉइस इनपुट के लिए Google Chrome या Microsoft Edge (और इंटरनेट) चाहिए। आप नीचे फ़ॉर्म से भी खोज सकते हैं।",
    formTitle: "आपकी जानकारी", reset: "रीसेट",
    fAge: "उम्र", fGender: "लिंग", fState: "राज्य / केंद्रशासित प्रदेश", fArea: "क्षेत्र",
    fCat: "सामाजिक वर्ग", fOcc: "व्यवसाय", fIncome: "वार्षिक पारिवारिक आय (₹)",
    fBpl: "बीपीएल / अंत्योदय राशन कार्ड", fDis: "दिव्यांगजन",
    fWid: "विधवा", fPreg: "गर्भवती / नई माता",
    oSelect: "— चुनें —", oFemale: "महिला", oMale: "पुरुष", oOther: "अन्य / ट्रांसजेंडर",
    oRural: "ग्रामीण (गांव)", oUrban: "शहरी (कस्बा/शहर)",
    oGen: "सामान्य", oObc: "ओबीसी", oSc: "अनुसूचित जाति", oSt: "अनुसूचित जनजाति", oEws: "ईडब्ल्यूएस", oMin: "अल्पसंख्यक",
    oStudent: "छात्र/छात्रा", oFarmer: "किसान", oWorker: "दिहाड़ी / निर्माण श्रमिक",
    oSalaried: "वेतनभोगी कर्मचारी", oSelf: "व्यवसाय / स्वरोजगार",
    oVendor: "रेहड़ी-पटरी विक्रेता", oArtisan: "कारीगर / शिल्पकार",
    oFisher: "मछुआरा", oHome: "गृहिणी", oUnemp: "बेरोजगार", oRetired: "सेवानिवृत्त",
    lakh: "लाख", cta: "मेरी योजनाएं खोजें",
    resultsFound: n => `आप <em>${n}</em> योजनाओं के पात्र हो सकते हैं`,
    resultsNone: "कोई मिलती-जुलती योजना नहीं मिली",
    noResultsMsg: "कम फ़ील्ड भरकर देखें, या आय की राशि जांचें। खाली छोड़ा गया फ़ील्ड “कोई भी” माना जाता है।",
    allTags: "सभी",
    verify: "MyScheme पर देखें →",
    ageChip: a => `उम्र: ${a}`, incomeChip: i => `आय: ₹${Number(i).toLocaleString("en-IN")}`,
    disclaimer: "योजना विवरण सांकेतिक हैं और बदल सकते हैं। पात्रता की पुष्टि व आवेदन हमेशा आधिकारिक पोर्टल — myscheme.gov.in या अपनी राज्य सरकार की वेबसाइट से करें।",
    footer: "हर भारतीय को उसके अधिकारों की जानकारी",
    levels: { C: "केंद्र सरकार", S: "राज्य सरकार", CSS: "केंद्र प्रायोजित" },
    tags: { agri: "कृषि", health: "स्वास्थ्य", edu: "शिक्षा", housing: "आवास व सुविधाएं",
      pension: "पेंशन", wcd: "महिला व बाल", skill: "कौशल व रोजगार", biz: "व्यवसाय व ऋण",
      ins: "बीमा", fin: "बचत व बैंकिंग", food: "खाद्य सुरक्षा", worker: "श्रमिक",
      social: "समाज कल्याण" },
    chips: { age: (a, b) => a != null && b != null ? `उम्र ${a}–${b}` : (a != null ? `उम्र ${a}+` : `${b} वर्ष तक`),
      inc: v => `आय ≤ ₹${Number(v).toLocaleString("en-IN")}`,
      F: "महिला", M: "पुरुष", O: "ट्रांसजेंडर", R: "ग्रामीण", U: "शहरी",
      bpl: "बीपीएल", dis: "दिव्यांगता", wid: "विधवा", preg: "मातृत्व",
      sow: "एससी/एसटी या महिला", allIndia: "अखिल भारतीय",
      cat: { sc: "एससी", st: "एसटी", obc: "ओबीसी", gen: "सामान्य", ews: "ईडब्ल्यूएस", minority: "अल्पसंख्यक" },
      occ: { student: "छात्र", farmer: "किसान", worker: "श्रमिक", salaried: "वेतनभोगी",
        self_employed: "स्वरोजगार", unemployed: "बेरोजगार", homemaker: "गृहिणी",
        fisher: "मछुआरे", artisan: "कारीगर", street_vendor: "रेहड़ी विक्रेता", retired: "सेवानिवृत्त" } },
  },
};

let LANG = "en";
const $ = id => document.getElementById(id);
const T = () => I18N[LANG];

function setLang(l) {
  LANG = l;
  $("btn-en").classList.toggle("active", l === "en");
  $("btn-hi").classList.toggle("active", l === "hi");
  document.documentElement.lang = l;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.dataset.i18n;
    if (typeof T()[key] === "string") el.textContent = T()[key];
  });
  fillStates();
  if (lastResults) renderResults(lastResults);
}

function fillStates() {
  const sel = $("f-state");
  const current = sel.value;
  sel.innerHTML = `<option value="">${T().oSelect}</option>` +
    STATES_LIST.map(([en, hi]) =>
      `<option value="${en}">${LANG === "hi" ? hi : en}</option>`).join("");
  sel.value = current;
}

/* ---------------- eligibility engine ---------------- */
function readProfile() {
  return {
    age: $("f-age").value ? +$("f-age").value : null,
    gender: $("f-gender").value || null,
    state: $("f-state").value || null,
    res: $("f-res").value || null,
    cat: $("f-cat").value || null,
    occ: $("f-occ").value || null,
    income: $("f-income").value !== "" ? +$("f-income").value : null,
    bpl: $("f-bpl").checked, dis: $("f-dis").checked,
    wid: $("f-wid").checked, preg: $("f-preg").checked,
  };
}

function matches(s, p) {
  const e = s.e || {};
  // State scoping: ALL-India schemes always shown; state schemes only for that state
  if (s.st !== "ALL" && p.state && s.st !== p.state) return false;
  if (s.st !== "ALL" && !p.state) return false;
  if (e.sts && p.state && !e.sts.includes(p.state)) return false;
  if (e.sts && !p.state) return false;

  if (p.age != null) {
    if (e.aMin != null && p.age < e.aMin) return false;
    if (e.aMax != null && p.age > e.aMax) return false;
  }
  if (e.g && p.gender && e.g !== p.gender) return false;
  if (e.inc != null && p.income != null && p.income > e.inc) return false;
  if (e.cat && p.cat && !e.cat.includes(p.cat)) return false;
  if (e.occ && p.occ && !e.occ.includes(p.occ)) return false;
  if (e.res && p.res && e.res !== p.res) return false;

  // Requirement flags: scheme only applies when the user has the attribute
  if (e.bpl && !p.bpl) return false;
  if (e.dis && !p.dis) return false;
  if (e.wid && !p.wid) return false;
  if (e.preg && !p.preg) return false;
  // Stand-Up India rule: SC/ST OR woman
  if (e.sow && !(p.gender === "F" || p.cat === "sc" || p.cat === "st")) return false;
  return true;
}

/* Rank: state-specific first, then more targeted (more constraints) first */
function rank(s, p) {
  let score = 0;
  if (s.lvl === "S") score += 30;
  if (s.st !== "ALL") score += 20;
  const e = s.e || {};
  ["g", "cat", "occ", "bpl", "dis", "wid", "preg", "inc", "aMin", "aMax"]
    .forEach(k => { if (e[k] != null && e[k] !== false) score += 3; });
  if (e.occ && p.occ && e.occ.includes(p.occ)) score += 8;
  return -score;
}

let lastResults = null;
let activeTag = "all";

function findSchemes() {
  const p = readProfile();
  const out = SCHEMES_DB.filter(s => matches(s, p)).sort((a, b) => rank(a, p) - rank(b, p));
  lastResults = out;
  activeTag = "all";
  renderResults(out);
  $("results-section").hidden = false;
  $("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

function eligChips(s) {
  const e = s.e || {}, c = T().chips, chips = [];
  chips.push(s.st === "ALL" ? c.allIndia : (LANG === "hi" ? stateHi(s.st) : s.st));
  if (e.aMin != null || e.aMax != null) chips.push(c.age(e.aMin ?? null, e.aMax ?? null));
  if (e.g) chips.push(c[e.g]);
  if (e.inc != null) chips.push(c.inc(e.inc));
  if (e.cat) chips.push(e.cat.map(x => c.cat[x]).join("/"));
  if (e.occ) chips.push(e.occ.slice(0, 3).map(x => c.occ[x]).join(", "));
  if (e.res) chips.push(c[e.res]);
  ["bpl", "dis", "wid", "preg"].forEach(k => { if (e[k]) chips.push(c[k]); });
  if (e.sow) chips.push(c.sow);
  return chips;
}

function stateHi(en) {
  const f = STATES_LIST.find(s => s[0] === en);
  return f ? f[1] : en;
}

function renderResults(list) {
  const shown = activeTag === "all" ? list : list.filter(s => s.tag === activeTag);
  const rc = $("results-count");
  rc.innerHTML = list.length ? T().resultsFound(list.length) : T().resultsNone;

  // tag filter buttons
  const tags = [...new Set(list.map(s => s.tag))];
  $("tag-filters").innerHTML =
    `<button class="tag-filter ${activeTag === "all" ? "active" : ""}" onclick="setTag('all')">${T().allTags} (${list.length})</button>` +
    tags.map(t => {
      const n = list.filter(s => s.tag === t).length;
      return `<button class="tag-filter ${activeTag === t ? "active" : ""}" onclick="setTag('${t}')">${T().tags[t]} (${n})</button>`;
    }).join("");

  const el = $("results");
  if (!list.length) {
    el.innerHTML = `<div class="no-results">${T().noResultsMsg}</div>`;
    return;
  }
  const cap = 400; // render cap for performance
  el.innerHTML = shown.slice(0, cap).map((s, i) => {
    const name = LANG === "hi" ? s.nh : s.n;
    const ben = LANG === "hi" ? s.bh : s.b;
    const q = encodeURIComponent(s.n.replace(/ – .*$/, ""));
    return `<article class="card" style="animation-delay:${Math.min(i, 20) * 0.03}s">
      <div class="card-top">
        <h4>${name}</h4>
        <span class="badge badge-${s.lvl}">${T().levels[s.lvl]}</span>
      </div>
      <p class="card-benefit">${ben}</p>
      <div class="card-meta">${eligChips(s).map(x => `<span class="mchip">${x}</span>`).join("")}</div>
      <a class="card-link" href="https://www.myscheme.gov.in/search?q=${q}" target="_blank" rel="noopener">${T().verify}</a>
    </article>`;
  }).join("");
}

function setTag(t) { activeTag = t; renderResults(lastResults || []); }

function setIncome(v) { $("f-income").value = v; }

function resetForm() {
  ["f-age", "f-income"].forEach(id => $(id).value = "");
  ["f-gender", "f-state", "f-res", "f-cat", "f-occ"].forEach(id => $(id).value = "");
  ["f-bpl", "f-dis", "f-wid", "f-preg"].forEach(id => $(id).checked = false);
  $("results-section").hidden = true;
  $("parsed-chips").innerHTML = "";
  $("transcript").hidden = true;
  lastResults = null;
}

/* ---------------- voice input ---------------- */
const VOICE_LANGS = [
  ["hi-IN", "हिन्दी (Hindi)"], ["en-IN", "English (India)"],
  ["bn-IN", "বাংলা (Bengali)"], ["ta-IN", "தமிழ் (Tamil)"],
  ["te-IN", "తెలుగు (Telugu)"], ["mr-IN", "मराठी (Marathi)"],
  ["gu-IN", "ગુજરાતી (Gujarati)"], ["kn-IN", "ಕನ್ನಡ (Kannada)"],
  ["ml-IN", "മലയാളം (Malayalam)"], ["pa-IN", "ਪੰਜਾਬੀ (Punjabi)"],
  ["ur-IN", "اردو (Urdu)"], ["or-IN", "ଓଡ଼ିଆ (Odia)"],
];

let recognition = null, listening = false;

function initVoice() {
  const sel = $("voice-lang");
  sel.innerHTML = VOICE_LANGS.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $("voice-unsupported").hidden = false; return; }
  recognition = new SR();
  recognition.continuous = false;
  recognition.interimResults = true;

  recognition.onresult = ev => {
    let txt = "";
    for (const r of ev.results) txt += r[0].transcript;
    $("transcript").hidden = false;
    $("transcript").textContent = "🎙 " + txt;
    if (ev.results[ev.results.length - 1].isFinal) {
      applyTranscript(txt);
    }
  };
  recognition.onend = () => setListening(false);
  recognition.onerror = ev => {
    setListening(false);
    if (ev.error === "not-allowed" || ev.error === "service-not-allowed" || ev.error === "network") {
      $("voice-unsupported").hidden = false;
    }
  };
}

function setListening(on) {
  listening = on;
  $("mic-btn").classList.toggle("listening", on);
  const st = $("voice-status");
  st.classList.toggle("active", on);
  st.textContent = on ? T().voiceListening : T().voiceIdle;
}

function toggleMic() {
  if (!recognition) { $("voice-unsupported").hidden = false; return; }
  if (listening) { recognition.stop(); return; }
  recognition.lang = $("voice-lang").value;
  try { recognition.start(); setListening(true); } catch (_) { /* already running */ }
}

/* ------------- transcript understanding (English + Hindi) ------------- */
const HI_DIGITS = { "०": "0", "१": "1", "२": "2", "३": "3", "४": "4", "५": "5", "६": "6", "७": "7", "८": "8", "९": "9" };
const HI_NUM_WORDS = {
  "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "सात": 7,
  "आठ": 8, "नौ": 9, "दस": 10, "बीस": 20, "पच्चीस": 25, "तीस": 30, "पैंतीस": 35,
  "चालीस": 40, "पैंतालीस": 45, "पचास": 50, "पचपन": 55, "साठ": 60, "पैंसठ": 65,
  "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
};

const OCC_WORDS = {
  farmer: ["farmer", "farming", "kisan", "kheti", "किसान", "खेती", "कृषक", "काश्तकार"],
  student: ["student", "study", "studying", "college", "school", "padh", "vidyarthi",
    "छात्र", "छात्रा", "विद्यार्थी", "पढ़", "पढ़ाई", "कॉलेज", "स्कूल"],
  worker: ["labour", "laborer", "labourer", "worker", "mazdoor", "construction", "daily wage",
    "मजदूर", "मज़दूर", "श्रमिक", "दिहाड़ी", "निर्माण"],
  street_vendor: ["vendor", "hawker", "thela", "rehri", "रेहड़ी", "ठेला", "पटरी", "फेरीवाला"],
  fisher: ["fisherman", "fisher", "machhuara", "मछुआरा", "मछली"],
  artisan: ["artisan", "carpenter", "weaver", "tailor", "blacksmith", "potter", "kaarigar",
    "कारीगर", "बढ़ई", "बुनकर", "दर्जी", "लोहार", "कुम्हार", "शिल्पकार"],
  self_employed: ["business", "shop", "dukaan", "vyapar", "entrepreneur", "self employed",
    "व्यापार", "व्यवसाय", "दुकान", "उद्यमी", "स्वरोजगार", "बिजनेस"],
  salaried: ["salaried", "job", "naukri", "employee", "नौकरी", "कर्मचारी", "वेतन"],
  unemployed: ["unemployed", "berozgar", "no job", "jobless", "बेरोजगार", "बेरोज़गार"],
  homemaker: ["housewife", "homemaker", "grihini", "गृहिणी", "घरेलू महिला"],
  retired: ["retired", "pension", "सेवानिवृत्त", "रिटायर"],
};

const CAT_WORDS = {
  sc: ["scheduled caste", " sc ", "dalit", "अनुसूचित जाति", "दलित", "एससी"],
  st: ["scheduled tribe", " st ", "tribal", "adivasi", "अनुसूचित जनजाति", "आदिवासी", "जनजाति", "एसटी"],
  obc: ["obc", "backward class", "pichhda", "ओबीसी", "पिछड़ा", "पिछड़ी"],
  minority: ["minority", "muslim", "christian", "sikh", "buddhist", "parsi", "jain",
    "अल्पसंख्यक", "मुस्लिम", "ईसाई", "सिख", "बौद्ध", "जैन", "पारसी"],
  ews: ["ews", "ईडब्ल्यूएस", "आर्थिक रूप से कमजोर"],
  gen: ["general category", "general caste", "सामान्य वर्ग", "जनरल"],
};

const STATE_ALIASES = {
  "उत्तर प्रदेश": "Uttar Pradesh", "यूपी": "Uttar Pradesh", "up": "Uttar Pradesh",
  "मध्य प्रदेश": "Madhya Pradesh", "एमपी": "Madhya Pradesh",
  "झारखंड": "Jharkhand", "झारखण्ड": "Jharkhand",
  "बिहार": "Bihar", "पंजाब": "Punjab", "हरियाणा": "Haryana", "गुजरात": "Gujarat",
  "राजस्थान": "Rajasthan", "महाराष्ट्र": "Maharashtra", "कर्नाटक": "Karnataka",
  "केरल": "Kerala", "तमिलनाडु": "Tamil Nadu", "तेलंगाना": "Telangana",
  "आंध्र प्रदेश": "Andhra Pradesh", "आंध्र": "Andhra Pradesh",
  "पश्चिम बंगाल": "West Bengal", "बंगाल": "West Bengal", "bengal": "West Bengal",
  "ओडिशा": "Odisha", "उड़ीसा": "Odisha", "orissa": "Odisha",
  "छत्तीसगढ़": "Chhattisgarh", "असम": "Assam", "दिल्ली": "Delhi",
  "उत्तराखंड": "Uttarakhand", "हिमाचल": "Himachal Pradesh", "हिमाचल प्रदेश": "Himachal Pradesh",
  "जम्मू": "Jammu and Kashmir", "कश्मीर": "Jammu and Kashmir", "jammu": "Jammu and Kashmir",
  "kashmir": "Jammu and Kashmir", "गोवा": "Goa", "त्रिपुरा": "Tripura", "मणिपुर": "Manipur",
  "मेघालय": "Meghalaya", "मिजोरम": "Mizoram", "मिज़ोरम": "Mizoram", "नागालैंड": "Nagaland",
  "सिक्किम": "Sikkim", "अरुणाचल": "Arunachal Pradesh", "अरुणाचल प्रदेश": "Arunachal Pradesh",
  "चंडीगढ़": "Chandigarh", "पुडुचेरी": "Puducherry", "pondicherry": "Puducherry",
  "लद्दाख": "Ladakh", "लक्षद्वीप": "Lakshadweep", "अंडमान": "Andaman and Nicobar Islands",
  "andaman": "Andaman and Nicobar Islands",
};

function normText(t) {
  let x = " " + t.toLowerCase() + " ";
  for (const [d, a] of Object.entries(HI_DIGITS)) x = x.split(d).join(a);
  return x;
}

function parseTranscript(raw) {
  const t = normText(raw);
  const p = {};

  // --- age: "35 years old", "35 साल", "उम्र 35", "age 35" ---
  let m = t.match(/(\d{1,3})\s*(?:years?|yrs?|saal|sal|varsh|साल|वर्ष|बरस)/);
  if (!m) m = t.match(/(?:age|umar|umr|उम्र|आयु)\s*(?:is|है|:)?\s*(\d{1,3})/);
  if (m && +m[1] > 0 && +m[1] < 111) p.age = +m[1];
  if (p.age == null) {
    for (const [w, v] of Object.entries(HI_NUM_WORDS)) {
      if (t.includes(w + " साल") || t.includes(w + " वर्ष")) { p.age = v; break; }
    }
  }

  // --- income: "2 lakh", "income 200000", "आय 2 लाख", "50 hazar" ---
  m = t.match(/(\d+(?:\.\d+)?)\s*(?:lakh|lac|लाख)/);
  if (m) p.income = Math.round(+m[1] * 100000);
  if (p.income == null) {
    m = t.match(/(\d+(?:\.\d+)?)\s*(?:hazaar|hazar|thousand|हजार|हज़ार)/);
    if (m) p.income = Math.round(+m[1] * 1000);
  }
  if (p.income == null) {
    m = t.match(/(?:income|aay|kamai|आय|आमदनी|कमाई)\s*(?:is|है|:)?\s*(?:₹|rs\.?|rupees?)?\s*(\d{4,9})/);
    if (m) p.income = +m[1];
  }

  // --- gender ---
  if (/\b(?:female|woman|girl|lady|mahila|aurat|ladki)\b/.test(t) ||
      /महिला|औरत|लड़की|स्त्री|विधवा|गर्भवती|गृहिणी|छात्रा/.test(t)) p.gender = "F";
  else if (/\b(?:male|man|boy|purush|aadmi|ladka)\b/.test(t) ||
      /पुरुष|आदमी|लड़का/.test(t)) p.gender = "M";
  if (/transgender|ट्रांसजेंडर|किन्नर/.test(t)) p.gender = "O";

  // --- state: full English names, then aliases ---
  for (const [en] of STATES_LIST) {
    if (t.includes(en.toLowerCase())) { p.state = en; break; }
  }
  if (!p.state) {
    for (const [alias, en] of Object.entries(STATE_ALIASES)) {
      const a = alias.toLowerCase();
      const hit = /^[a-z]+$/.test(a) ? new RegExp("\\b" + a + "\\b").test(t) : t.includes(a);
      if (hit) { p.state = en; break; }
    }
  }

  // --- residence ---
  if (/village|gaon|gram|rural|गांव|गाँव|ग्रामीण|देहात/.test(t)) p.res = "R";
  else if (/city|town|shahar|urban|शहर|शहरी|नगर/.test(t)) p.res = "U";

  // --- occupation ---
  outer:
  for (const [k, words] of Object.entries(OCC_WORDS)) {
    for (const w of words) {
      const hit = /^[a-z ]+$/.test(w) ? t.includes(w) : t.includes(w);
      if (hit) { p.occ = k; break outer; }
    }
  }

  // --- category ---
  for (const [k, words] of Object.entries(CAT_WORDS)) {
    if (words.some(w => t.includes(w))) { p.cat = k; break; }
  }

  // --- flags ---
  if (/widow|विधवा/.test(t)) { p.wid = true; p.gender = "F"; }
  if (/disab|handicap|divyang|viklang|दिव्यांग|विकलांग/.test(t)) p.dis = true;
  if (/\bbpl\b|ration card|garibi|below poverty|बीपीएल|गरीबी रेखा|राशन कार्ड|अंत्योदय/.test(t)) p.bpl = true;
  if (/pregnant|garbhvati|गर्भवती|प्रेग्नेंट/.test(t)) { p.preg = true; p.gender = "F"; }

  return p;
}

function applyTranscript(txt) {
  const p = parseTranscript(txt);
  const chips = [];
  const c = T();
  if (p.age != null) { $("f-age").value = p.age; chips.push(c.ageChip(p.age)); }
  if (p.gender) { $("f-gender").value = p.gender; chips.push(c.chips[p.gender]); }
  if (p.state) { $("f-state").value = p.state; chips.push(LANG === "hi" ? stateHi(p.state) : p.state); }
  if (p.res) { $("f-res").value = p.res; chips.push(c.chips[p.res]); }
  if (p.cat) { $("f-cat").value = p.cat; chips.push(c.chips.cat[p.cat]); }
  if (p.occ) { $("f-occ").value = p.occ; chips.push(c.chips.occ[p.occ]); }
  if (p.income != null) { $("f-income").value = p.income; chips.push(c.incomeChip(p.income)); }
  ["bpl", "dis", "wid", "preg"].forEach(k => {
    if (p[k]) { $("f-" + k).checked = true; chips.push(c.chips[k]); }
  });
  $("parsed-chips").innerHTML = chips.map(x => `<span class="pchip">✓ ${x}</span>`).join("");
  if (chips.length) findSchemes();
}

/* ---------------- heartbeat for the desktop launcher ---------------- */
function startHeartbeat() {
  if (location.protocol !== "http:") return;
  setInterval(() => { fetch("/ping").catch(() => {}); }, 5000);
}

/* ---------------- init ---------------- */
window.addEventListener("DOMContentLoaded", () => {
  $("stat-schemes").textContent = SCHEMES_DB.length.toLocaleString("en-IN");
  fillStates();
  initVoice();
  startHeartbeat();
});
