import base64
import io
import json
import re
import sqlite3
import time
import unicodedata
from datetime import datetime
from difflib import get_close_matches
from html import escape
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

DB, GH_FILE = "cantina.db", "cantina_data.json"
TABLES = ["producers", "styles", "shops", "avail"]
KNOWN_COUNTRIES = ["Francia", "Italia", "Spagna", "Germania", "Austria", "Portogallo", "Svizzera", "Slovenia", "Grecia", "Ungheria", "USA"]
BASE_TYPES = ["Bianco", "Rosso", "Rosato", "Bolla", "Arancione", "Dolce"]
STOP = {"domaine", "chateau", "maison", "tenuta", "cantina", "azienda", "agricola", "weingut", "bodega", "bodegas", "fattoria",
        "podere", "castello", "marchesi", "famiglia", "vini", "winery", "estate", "des", "les", "del", "della", "dei", "von"}
# Siti che conosco già: (paese, indirizzo di ricerca con {q}). Gli altri li completi nella scheda Siti.
KNOWN_SHOPS = {
    "vinossimo": ("Francia", "https://vinossimo.com/recherche?controller=search&s={q}"),
    "versus": ("", "https://www.versus.wine/catalogue?searchQuery={q}"),
    "caves carrier": ("Francia", "https://www.caves-carriere.fr/recherche?s={q}"),
    "buveur de vin": ("Francia", "https://buveurdevin.com/?s={q}&post_type=product"),
    "wine rl": ("Austria", "https://www.winerl.com/en/#8b1f/mobile/m=and&q={q}"),
    "weinfurore": ("Germania", "https://www.weinfurore.de/?qs={q}"),
    "kierdorfwein": ("Germania", "https://www.kierdorfwein.de/search?keywords={q}&page=search&page_action=query"),
    "restaurant cave la pepiniere": ("Francia", "https://restaurant-cave-lapepiniere.com/search?q={q}&type=product"),
}


# ───────────────────────── Database ─────────────────────────
def db():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys=ON")
    return con


def q(sql, p=()):
    con = db()
    try:
        df = pd.read_sql_query(sql, con, params=p)
    finally:
        con.close()
    for c in df.columns:  # con pandas recente i testi vuoti (NULL) diventano NaN, che in Python vale «vero»: li riporto a None
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def init():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS producers(id INTEGER PRIMARY KEY, name TEXT UNIQUE, country TEXT, region TEXT, subregion TEXT, notes TEXT);
        CREATE TABLE IF NOT EXISTS styles(id INTEGER PRIMARY KEY, producer_id INTEGER REFERENCES producers(id) ON DELETE CASCADE, type TEXT, grape TEXT, wine TEXT);
        CREATE TABLE IF NOT EXISTS shops(id INTEGER PRIMARY KEY, name TEXT UNIQUE, country TEXT, url TEXT);
        CREATE TABLE IF NOT EXISTS avail(producer_id INTEGER REFERENCES producers(id) ON DELETE CASCADE,
                                         shop_id INTEGER REFERENCES shops(id) ON DELETE CASCADE, PRIMARY KEY(producer_id, shop_id));
        """)
        for table, col in (("producers", "subregion"), ("styles", "wine")):  # database creati con la versione precedente
            if col not in [r[1] for r in c.execute(f"PRAGMA table_info({table})")]:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")


def secret(k, d=None):
    try:
        return st.secrets.get(k, d)
    except Exception:
        return d


# ───────────────────────── Compilazione automatica (Gemini o Claude) ─────────────────────────
def parse_json(text):
    """Legge il primo oggetto JSON nel testo (ignora recinzioni ``` e commenti prima o dopo)."""
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


GEM = "https://generativelanguage.googleapis.com/v1beta"


def gemini_candidates():
    """Modelli da provare: GEMINI_MODEL se impostato, altrimenti quelli 'flash' disponibili per la tua chiave."""
    if secret("GEMINI_MODEL"):
        return [secret("GEMINI_MODEL")]
    r = requests.get(f"{GEM}/models?pageSize=200", headers={"x-goog-api-key": secret("GEMINI_API_KEY")}, timeout=20)
    r.raise_for_status()
    names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    bad = ("image", "tts", "live", "audio", "embedding", "robotics", "computer", "native", "thinking", "exp", "deep")
    cand = [n for n in names if n.startswith("gemini-") and "flash" in n and not any(x in n for x in bad)]

    def rank(n):
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (not n.endswith("-latest"), "preview" in n, "lite" in n, -(float(v.group(1)) if v else 0))
    return sorted(cand, key=rank)[:4]


def gemini_call(prompt, grounded=False):
    """Ritorna (testo, fonti). grounded=True attiva la ricerca Google di Gemini e restituisce le pagine consultate."""
    last, fixed = "nessun modello 'flash' trovato per questa chiave", bool(secret("GEMINI_MODEL"))
    try:
        models = gemini_candidates()
    except requests.exceptions.RequestException:
        raise RuntimeError("Google non risponde in questo momento: riprova tra un minuto.")
    for model in (models[:2] if grounded else models):
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        if grounded:
            body["tools"] = [{"google_search": {}}]
        else:
            body["generationConfig"] = {"responseMimeType": "application/json"}
        r = None
        for attempt in range(2):  # un secondo tentativo se Google è momentaneamente sovraccarico
            try:
                r = requests.post(f"{GEM}/models/{model}:generateContent",
                                  headers={"x-goog-api-key": secret("GEMINI_API_KEY"), "Content-Type": "application/json"},
                                  json=body, timeout=(10, 60 if grounded else 30))
            except requests.exceptions.RequestException:  # nessuna risposta in tempo: passo al modello successivo
                r = None
                break
            if r.status_code in (500, 502, 503, 504) and attempt == 0:
                time.sleep(2)
                continue
            break
        if r is None or r.status_code in (403, 404, 429, 500, 502, 503, 504):
            last = f"{model}: " + ("nessuna risposta in tempo" if r is None else f"errore {r.status_code}")
            if fixed:
                break
            continue
        r.raise_for_status()
        cand = r.json()["candidates"][0]
        text = "".join(p.get("text", "") for p in cand["content"]["parts"])
        chunks = cand.get("groundingMetadata", {}).get("groundingChunks", [])
        sources = [{"title": c["web"].get("title", ""), "uri": c["web"]["uri"]} for c in chunks if c.get("web", {}).get("uri")]
        return text, sources
    raise RuntimeError(f"Gemini non ha risposto ({last}). Riprova tra un minuto; "
                       "altrimenti usa «Incolla i dati da una chat» oppure controlla chiave e modello nei Secrets.")


def snap_to(value, pool, cutoff=0.85):
    """Riporta un valore a quello già in uso nella rubrica (stessa grafia), se è molto simile."""
    v, m = " ".join(str(value or "").split()), {p.lower(): p for p in pool if p}
    hit = get_close_matches(v.lower(), list(m), 1, cutoff)
    return m[hit[0]] if hit else v


def latin(v):
    return all(unicodedata.name(ch, "").startswith("LATIN") for ch in v if ch.isalpha())


ALIAS = {"france": "Francia", "italy": "Italia", "spain": "Spagna", "germany": "Germania", "deutschland": "Germania", "osterreich": "Austria",
         "loire": "Loira", "loire valley": "Loira", "burgundy": "Borgogna", "bourgogne": "Borgogna", "rhone": "Rodano", "rhone valley": "Rodano",
         "tuscany": "Toscana", "piedmont": "Piemonte", "sicily": "Sicilia", "sardinia": "Sardegna", "lombardy": "Lombardia",
         "apulia": "Puglia", "south tyrol": "Alto Adige", "sudtirol": "Alto Adige", "styria": "Steiermark", "steiermark": "Steiermark",
         "lower austria": "Niederösterreich", "niederosterreich": "Niederösterreich", "canary islands": "Canarie", "basque country": "Paesi Baschi"}


def norm_type(t, types):
    c = norm(t)
    for key, val in (("bianco", "Bianco"), ("white", "Bianco"), ("blanc", "Bianco"), ("weiss", "Bianco"),
                     ("rosso", "Rosso"), ("red", "Rosso"), ("rouge", "Rosso"), ("rot", "Rosso"), ("rosato", "Rosato"), ("rose", "Rosato"),
                     ("bolla", "Bolla"), ("spumante", "Bolla"), ("sparkling", "Bolla"), ("champagne", "Bolla"), ("mousseux", "Bolla"),
                     ("cremant", "Bolla"), ("sekt", "Bolla"), ("bulles", "Bolla"), ("arancio", "Arancione"), ("orange", "Arancione"),
                     ("dolce", "Dolce"), ("sweet", "Dolce"), ("liquoroso", "Dolce")):
        if c.startswith(key):
            return val
    return snap_to(t, types)


def autofill_prompt(producer, opts, web=False):
    lists = (f"Paesi già in uso: {', '.join(opts['countries'][:30])}. Regioni già in uso: {', '.join(opts['regions'][:60])}. "
             f"Uve già in uso: {', '.join(opts['grapes'][:60])}. ")
    return (f"""Sei un sommelier esperto. Per il produttore di vino "{producer}" rispondi SOLO con JSON valido, senza altro testo:
{{"country":"","region":"","subregion":"","styles":[{{"type":"Bianco|Rosso|Rosato|Bolla|Arancione|Dolce","grape":""}}]}}
""" + ("Usa la ricerca web (sito ufficiale, guide, negozi specializzati) per verificare le informazioni. " if web else "") + f"""
- country e region: in italiano dove esiste la forma italiana (es. Francia, Borgogna, Loira, Toscana). {lists}Se un valore coincide con uno già in uso, scrivilo identico.
- subregion: la provincia o sotto-regione/denominazione principale, nel nome originale (es. Saumur, Südsteiermark, Côte de Nuits).
- styles: i tipi di vino e le uve principali che il produttore lavora davvero (max 8 coppie), una coppia tipo+uva per riga, con il nome comune dell'uva (es. Chenin Blanc, Pinot Nero).
- Includi solo ciò di cui sei certo: se non conosci il produttore rispondi {{"error":"sconosciuto"}}.""")


def autofill(producer, opts):
    """Ritorna un dizionario con country, region, subregion, styles e '_sources' (pagine web consultate), oppure None se manca la chiave."""
    if secret("ANTHROPIC_API_KEY"):
        import anthropic
        r = anthropic.Anthropic(api_key=secret("ANTHROPIC_API_KEY")).messages.create(
            model=secret("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=1500,
            messages=[{"role": "user", "content": autofill_prompt(producer, opts)}])
        return {**parse_json(r.content[0].text), "_sources": []}
    if secret("GEMINI_API_KEY"):
        res, sources = None, []
        try:
            text, sources = gemini_call(autofill_prompt(producer, opts, web=True), grounded=True)
            res = parse_json(text)
        except Exception:
            res, sources = None, []
        if res is None:
            res = parse_json(gemini_call(autofill_prompt(producer, opts))[0])
        return {**res, "_sources": sources}
    return None


def clean_suggestion(res, opts):
    """Allinea la proposta ai valori già usati (paesi, regioni, tipi, uve) e scarta nomi con alfabeti non latini."""
    pairs, seen = [], set()
    for x in (res.get("styles") or [])[:12]:
        t, g = norm_type(x.get("type", ""), opts["types"]), snap_to(x.get("grape", ""), opts["grapes"], 0.88)
        if (t or g) and latin(t + g) and (norm(t), norm(g)) not in seen:
            seen.add((norm(t), norm(g)))
            pairs.append((t, g))
    al = lambda v: ALIAS.get(norm(v), " ".join(str(v or "").split()))
    return {"country": snap_to(al(res.get("country")), opts["countries"], 0.8), "region": snap_to(al(res.get("region")), opts["regions"], 0.8),
            "subregion": snap_to(res.get("subregion"), opts["subs"], 0.9), "styles": pairs, "sources": res.get("_sources", [])}


def merge_suggestion(snap, sug, overwrite):
    """Applica la proposta ai campi del modulo: di norma solo ai campi vuoti e aggiungendo i tipi/uve mancanti.
    Se la proposta contraddice paese o regione già inseriti non applica nulla. Ritorna (campi, avviso)."""
    out = dict(snap)
    if not overwrite:
        for k, lab in (("country", "il paese"), ("region", "la regione")):
            if snap[k] and sug[k] and norm(snap[k]) != norm(sug[k]):
                return out, (f"La proposta indica {lab} «{sug[k]}», diverso da «{snap[k]}» che hai già inserito: non ho applicato nulla. "
                             "Se vuoi usarla, spunta «Sostituisci anche i campi già compilati» e riprova.")
    for k in ("country", "region", "subregion"):
        if sug[k] and (overwrite or not snap[k]):
            out[k] = sug[k]
    rows = [r for r in snap["styles"] if any(r)]
    if overwrite:
        rows = [r for r in rows if r[2]]  # restano solo le righe con un nome di vino scritto da te
    have = {(norm(t), norm(g)) for t, g, _ in rows}
    out["styles"] = rows + [(t, g, "") for t, g in sug["styles"] if (norm(t), norm(g)) not in have]
    return out, ""


# ───────────────────────── Salvataggio permanente (repository GitHub privato) ─────────────────────────
def dump():
    return {t: q(f"SELECT * FROM {t}").to_dict("records") for t in TABLES}


def restore(data):
    with db() as c:
        for t in reversed(TABLES):
            c.execute(f"DELETE FROM {t}")
        for t in TABLES:
            df = pd.DataFrame(data.get(t, []))
            if not df.empty:
                df.to_sql(t, c, if_exists="append", index=False)


def gh_conf():
    tok, repo = secret("GITHUB_TOKEN"), secret("GITHUB_REPO")
    return (tok, repo) if tok and repo else None


def gh_call(method, accept="application/vnd.github+json", **kw):
    tok, repo = gh_conf()
    h = {"Authorization": f"Bearer {tok}", "Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    return requests.request(method, f"https://api.github.com/repos/{repo}/contents/{GH_FILE}", headers=h, timeout=30, **kw)


def pull_backup():
    if not gh_conf():
        return False
    try:
        r = gh_call("GET", accept="application/vnd.github.raw+json")
        if r.status_code != 200:
            return False
        restore(r.json())
        return True
    except Exception:
        return False


def push_backup():
    if not gh_conf():
        return None
    try:
        m = gh_call("GET")
        body = {"message": "backup " + datetime.now().strftime("%Y-%m-%d %H:%M"),
                "content": base64.b64encode(json.dumps(dump(), ensure_ascii=False, default=str).encode()).decode()}
        if m.status_code == 200 and m.json().get("sha"):
            body["sha"] = m.json()["sha"]
        return gh_call("PUT", json=body).status_code in (200, 201)
    except Exception:
        return False


def autosave():
    if push_backup() is False:
        st.toast("⚠️ Salvataggio su GitHub non riuscito: scarica un backup dalla scheda Dati.")


@st.cache_resource
def first_boot():
    """Una volta per avvio: se il database è vuoto, ripristina i dati da GitHub."""
    return q("SELECT 1 FROM producers LIMIT 1").empty and pull_backup()


# ───────────────────────── Dati ─────────────────────────
def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "").lower()).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def pkey(name):
    """Chiave per riconoscere lo stesso produttore (ignora maiuscole, accenti, ordine e parole come Domaine/Tenuta)."""
    return " ".join(sorted(t for t in norm(name).split() if len(t) > 2 and t not in STOP)) or norm(name)


def load_all():
    """Produttori con 'styles' = elenco di (tipo, uva, nome del vino) e 'sites' = siti assegnati."""
    prods = q("SELECT id,name,country,region,subregion,notes FROM producers ORDER BY name COLLATE NOCASE").to_dict("records")
    by = {p["id"]: {**p, **{k: p[k] or "" for k in ("country", "region", "subregion", "notes")}, "styles": [], "sites": []} for p in prods}
    for r in q("SELECT producer_id,type,grape,wine FROM styles ORDER BY id").to_dict("records"):
        by[r["producer_id"]]["styles"].append((r["type"] or "", r["grape"] or "", r["wine"] or ""))
    for r in q("SELECT a.producer_id, s.name FROM avail a JOIN shops s ON s.id=a.shop_id ORDER BY s.name COLLATE NOCASE").to_dict("records"):
        by[r["producer_id"]]["sites"].append(r["name"])
    return list(by.values())


def zone_of(p):
    return " · ".join(x for x in (p["country"], p["region"], p["subregion"]) if isinstance(x, str) and x)


def styles_text(p, with_wine=False):
    return ", ".join(" ".join(x for x in ((w if with_wine else ""), t, g) if x) for t, g, w in p["styles"] if t or g or (with_wine and w))


def find_producer(name, allp):
    k = pkey(name)
    return next((p for p in allp if pkey(p["name"]) == k), None)


def save_producer(pid, name, country, region, subregion, notes, styles, sites, merge_into=None):
    """Nuovo: crea. Esistente (pid): sostituisce. merge_into: unisce a un produttore già presente senza togliere nulla.
    styles = elenco di (tipo, uva, nome del vino)."""
    styles = [(t.strip(), g.strip(), w.strip()) for t, g, w in styles if t.strip() or g.strip() or w.strip()]
    with db() as c:
        if merge_into:
            pid = merge_into
            cur = c.execute("SELECT country, region, subregion, notes FROM producers WHERE id=?", (pid,)).fetchone()
            c.execute("UPDATE producers SET country=?, region=?, subregion=?, notes=? WHERE id=?",
                      (cur[0] or country, cur[1] or region, cur[2] or subregion, cur[3] or notes, pid))
            have = {(norm(t), norm(g), norm(w)) for t, g, w in c.execute("SELECT type, grape, wine FROM styles WHERE producer_id=?", (pid,))}
            styles = [x for x in styles if tuple(norm(v) for v in x) not in have]
        elif pid is None:
            pid = c.execute("INSERT INTO producers(name,country,region,subregion,notes) VALUES(?,?,?,?,?)",
                            (name, country, region, subregion, notes)).lastrowid
        else:
            c.execute("UPDATE producers SET name=?, country=?, region=?, subregion=?, notes=? WHERE id=?",
                      (name, country, region, subregion, notes, pid))
            c.execute("DELETE FROM styles WHERE producer_id=?", (pid,))
            c.execute("DELETE FROM avail WHERE producer_id=?", (pid,))
        c.executemany("INSERT INTO styles(producer_id,type,grape,wine) VALUES(?,?,?,?)", [(pid, *x) for x in styles])
        for s in sites:
            row = c.execute("SELECT id FROM shops WHERE name=?", (s,)).fetchone()
            if row:
                c.execute("INSERT OR IGNORE INTO avail(producer_id, shop_id) VALUES(?,?)", (pid, row[0]))
    return pid


def delete_producer(pid):
    with db() as c:
        c.execute("DELETE FROM producers WHERE id=?", (int(pid),))
    autosave()


def upsert_shop(name, country="", url=""):
    known = KNOWN_SHOPS.get(norm(name), ("", ""))
    with db() as c:
        c.execute("INSERT INTO shops(name,country,url) VALUES(?,?,?) ON CONFLICT(name) DO NOTHING",
                  (name, country or known[0], url or known[1]))


# ───────────────────────── Importazione dal vecchio foglio Excel ─────────────────────────
def _cl(x):
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else " ".join(str(x).split())


def read_old_xlsx(raw):
    """Legge il foglio con colonne Paese, Regione, Produttore, Tipo n, Uva n, Sito n. Ritorna (righe, siti, correzioni)."""
    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None, dtype=str)
    found = None
    for name, df in sheets.items():
        for i in range(min(10, len(df))):
            if "produttore" in [norm(x) for x in df.iloc[i].tolist()]:
                found = (name, df, i)
                break
        if found:
            break
    if not found:
        return None, None, ["Non trovo la colonna «Produttore» nel file."]
    sheet, df, hr = found
    header = [norm(x) for x in df.iloc[hr]]
    idx = lambda pat: {int(re.findall(r"\d+", h)[0]) if re.search(r"\d", h) else 0: i for i, h in enumerate(header) if re.fullmatch(pat, h)}
    col_c, col_r, col_p = idx("paese").get(0), idx("regione").get(0), idx("produttore").get(0)
    col_s = idx(r"sotto ?regione|subregione|provincia").get(0)
    types, grapes, sites_c = idx(r"tipo \d+"), idx(r"uva \d+"), idx(r"sito \d+")
    fixes, canon_sites, canon_regions = {}, [], []
    note = lambda k: fixes.__setitem__(k, fixes.get(k, 0) + 1)
    for name, d in sheets.items():  # liste dei menu a tendina (foglio «Elenchi»)
        hdr = [norm(x) for x in d.iloc[0]] if len(d) else []
        if "sito" in hdr and name != sheet:
            canon_sites = [_cl(x) for x in d.iloc[1:, hdr.index("sito")] if _cl(x)]
        if "regione" in hdr and name != sheet:
            canon_regions = [_cl(x) for x in d.iloc[1:, hdr.index("regione")] if _cl(x)]
    canon_sites = [re.sub(r"\bResturant\b", "Restaurant", s) for s in canon_sites]
    lm = {c.lower(): c for c in KNOWN_COUNTRIES}

    def clean(raw_val, kind, canon=None, cutoff=0.8):
        v = _cl(raw_val)
        if v != ("" if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)) else str(raw_val)):
            note("spazi superflui rimossi")
        if kind == "site":
            v = re.sub(r"\bResturant\b", "Restaurant", v)
        pool = {c.lower(): c for c in (canon or [])}
        m = get_close_matches(v.lower(), list(pool), 1, cutoff)
        if m and pool[m[0]] != v:
            note(f"«{v}» → «{pool[m[0]]}»")
            v = pool[m[0]]
        return v

    rows, by_key, used = [], {}, []
    for _, r in df.iloc[hr + 1:].iterrows():
        name = _cl(r.iloc[col_p]) if col_p is not None else ""
        if not name:
            continue
        country = clean(r.iloc[col_c], "c", KNOWN_COUNTRIES, 0.75) if col_c is not None else ""
        region = clean(r.iloc[col_r], "r", canon_regions, 0.85) if col_r is not None else ""
        sub = _cl(r.iloc[col_s]) if col_s is not None else ""
        styles = []
        for n, ti in sorted(types.items()):
            t, g = _cl(r.iloc[ti]).capitalize(), _cl(r.iloc[grapes[n]]) if n in grapes else ""
            if t or g:
                styles.append((t, g, ""))
        sites = [s for s in (clean(r.iloc[i], "site", canon_sites, 0.85) for _, i in sorted(sites_c.items())) if s]
        k = pkey(name)
        if k in by_key:  # stesso produttore inserito due volte: unisco
            p = by_key[k]
            p["styles"] += [x for x in styles if x not in p["styles"]]
            p["sites"] += [x for x in sites if x not in p["sites"]]
            p["country"], p["region"], p["subregion"] = p["country"] or country, p["region"] or region, p["subregion"] or sub
            note(f"produttore doppio unito: {p['name']}")
        else:
            by_key[k] = {"name": name, "country": country, "region": region, "subregion": sub, "styles": styles, "sites": sites}
            rows.append(by_key[k])
        used += sites
    shops = list(dict.fromkeys(canon_sites + used))
    return rows, shops, [f"{k} ({v})" if v > 1 else k for k, v in fixes.items()]


def import_rows(rows, shops):
    for s in shops:
        upsert_shop(s)
    allp = load_all()
    added = merged = 0
    for p in rows:
        ex = find_producer(p["name"], allp)
        save_producer(None, p["name"], p["country"], p["region"], p["subregion"], "", p["styles"], p["sites"], merge_into=ex["id"] if ex else None)
        merged, added = merged + bool(ex), added + (not ex)
        allp = load_all() if not ex else allp
    return added, merged


def export_df():
    allp = load_all()
    ns, nt = max([len(p["sites"]) for p in allp] + [3]), max([len(p["styles"]) for p in allp] + [2])
    out = []
    for p in allp:
        row = {"Paese": p["country"], "Regione": p["region"], "Sottoregione": p["subregion"], "Produttore": p["name"]}
        for i in range(nt):
            t, g, w = p["styles"][i] if i < len(p["styles"]) else ("", "", "")
            row[f"Tipo {i + 1}"], row[f"Uva {i + 1}"], row[f"Vino {i + 1}"] = t, g, w
        for i in range(ns):
            row[f"Sito {i + 1}"] = p["sites"][i] if i < len(p["sites"]) else ""
        row["Note"] = p["notes"]
        out.append(row)
    return pd.DataFrame(out)


# ───────────────────────── Schede ─────────────────────────
FLAGS = {"Francia": "🇫🇷", "Italia": "🇮🇹", "Spagna": "🇪🇸", "Germania": "🇩🇪", "Austria": "🇦🇹", "Portogallo": "🇵🇹", "Svizzera": "🇨🇭",
         "Slovenia": "🇸🇮", "Grecia": "🇬🇷", "Ungheria": "🇭🇺", "USA": "🇺🇸"}


def flagged_zone(p):
    return f"{FLAGS.get(p['country'], '')} {zone_of(p)}".strip()


def type_class(t):
    """Classe di colore per il tipo di vino (bianco, rosso, rosato, bolla, arancione, dolce)."""
    c = norm(t)
    return next((k for k in ("bianco", "rosso", "rosato", "bolla", "arancione", "dolce") if c.startswith(k[:4])), "")


def shop_link(shop, text):
    url = (shop or {}).get("url")
    if not isinstance(url, str) or not url:
        return ""
    href = url.replace("{q}", quote_plus(text))
    if not href.startswith(("http://", "https://")):
        return ""
    label = "Cerca" if "{q}" in url else "Apri sito"
    return f'<a class="wh-btn" href="{escape(href, quote=True)}" target="_blank" rel="noopener noreferrer">{label} ↗</a>'


def shop_home(shop):
    url = (shop or {}).get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return ""
    return "/".join(url.split("/")[:3])


def pcard(p, shop):
    zone = flagged_zone(p) or "Zona non indicata"
    chips = "".join(f'<span class="chip t-{type_class(t)}">{escape(t)}{" · " + escape(g) if t and g else escape(g)}</span>' for t, g, _ in p["styles"] if t or g)
    stripe = next((type_class(t) for t, _, _ in p["styles"] if type_class(t)), "none")
    others = [x for x in p["sites"] if not shop or x != shop["name"]]
    also = f'<span class="wh-also">Anche su: {escape(", ".join(others))}</span>' if others else '<span class="wh-also"></span>'
    return (f'<div class="wh-card s-{stripe}"><div class="wh-name">{escape(p["name"])}</div><div class="wh-zone">{escape(zone)}</div>'
            f'<div class="wh-tags">{chips}</div><div class="wh-foot">{also}{shop_link(shop, p["name"])}</div></div>')


def cards_html(items, shop):
    """Schede raggruppate per regione, con un'intestazione per ogni gruppo."""
    items = sorted(items, key=lambda p: (p["country"].lower(), p["region"].lower(), p["name"].lower()))
    counts = {}
    for p in items:
        counts[(p["country"], p["region"])] = counts.get((p["country"], p["region"]), 0) + 1
    out, last = [], None
    for p in items:
        g = (p["country"], p["region"])
        if g != last:
            label = p["region"] or p["country"] or "Zona non indicata"
            out.append(f'<div class="wh-group">{FLAGS.get(p["country"], "")} {escape(label)}<small>{counts[g]}</small></div>')
            last = g
        out.append(pcard(p, shop))
    return '<div class="wh-grid">' + "".join(out) + "</div>"


CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&display=swap');
:root{--wine:#A6264A;--wine-d:#7E1B38;--wine-dd:#4E0F23;--line:rgba(127,127,127,.26);--soft:rgba(127,127,127,.08)}
.block-container{padding-top:1.2rem;padding-bottom:4rem;max-width:1080px}
@media (max-width:640px){.block-container{padding-left:.75rem;padding-right:.75rem}}
footer{visibility:hidden}
h1,h2,h3{font-family:'Playfair Display',Georgia,'Times New Roman',serif}
.wh-hero{background:linear-gradient(135deg,var(--wine-dd) 0%,var(--wine-d) 55%,var(--wine) 100%);color:#fff;border-radius:18px;padding:1.1rem 1.2rem 1rem;margin:0 0 .9rem;box-shadow:0 6px 18px rgba(78,15,35,.25)}
.wh-hero h1{margin:0!important;padding:0!important;font-size:1.9rem!important;line-height:1.1!important;color:#fff!important}
.wh-hero p{margin:.25rem 0 .8rem;opacity:.85;font-size:.92rem;color:#fff}
.wh-kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem}
.wh-kpi div{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.22);border-radius:12px;padding:.45rem .5rem;text-align:center}
.wh-kpi b{display:block;font-family:'Playfair Display',Georgia,serif;font-size:1.4rem;color:#fff;line-height:1.1}
.wh-kpi span{font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:#fff;opacity:.85}
.wh-sitehead{display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap;margin:.7rem 0 .2rem}
.wh-sitehead h2{margin:0!important;padding:0!important;font-size:1.55rem!important;color:var(--wine)}
.wh-sitehead span{font-size:.85rem;opacity:.7}
.wh-sitehead a{color:var(--wine);font-weight:600;text-decoration:none;font-size:.85rem}
.wh-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:.75rem}
.wh-group{grid-column:1/-1;display:flex;align-items:center;gap:.5rem;margin:.7rem 0 .05rem;font-family:'Playfair Display',Georgia,serif;font-size:1.05rem;color:var(--wine)}
.wh-group:after{content:"";flex:1;height:1px;background:var(--line)}
.wh-group small{font-family:system-ui,sans-serif;font-size:.7rem;background:var(--soft);border:1px solid var(--line);border-radius:99px;padding:0 .45rem;color:inherit;opacity:.8}
.wh-card{position:relative;background:var(--soft);border:1px solid var(--line);border-radius:16px;padding:.8rem .95rem .75rem 1.1rem;box-shadow:0 1px 3px rgba(0,0,0,.06);overflow:hidden}
.wh-card:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--wine)}
.s-bianco:before{background:#D9B44A}.s-rosso:before{background:#A6264A}.s-rosato:before{background:#E88AA8}
.s-bolla:before{background:#7DC1CB}.s-arancione:before{background:#E08A2E}.s-dolce:before{background:#C79A3E}
.wh-name{font-family:'Playfair Display',Georgia,serif;font-size:1.12rem;font-weight:600;line-height:1.25}
.wh-zone{font-size:.8rem;opacity:.72;margin:.1rem 0 .45rem}
.chip{display:inline-block;font-size:.72rem;border:1px solid var(--line);border-radius:99px;padding:.08rem .55rem;margin:0 .3rem .25rem 0}
.t-bianco{background:rgba(217,180,74,.20);border-color:rgba(217,180,74,.6)}
.t-rosso{background:rgba(166,38,74,.16);border-color:rgba(166,38,74,.5)}
.t-rosato{background:rgba(232,138,168,.22);border-color:rgba(232,138,168,.6)}
.t-bolla{background:rgba(125,193,203,.20);border-color:rgba(125,193,203,.6)}
.t-arancione{background:rgba(224,138,46,.20);border-color:rgba(224,138,46,.6)}
.t-dolce{background:rgba(199,154,62,.22);border-color:rgba(199,154,62,.6)}
.wh-foot{display:flex;justify-content:space-between;align-items:center;gap:.6rem;margin-top:.3rem}
.wh-also{font-size:.74rem;opacity:.65;flex:1}
.wh-btn{display:inline-block;background:var(--wine);color:#fff!important;border-radius:99px;padding:.28rem .85rem;font-size:.8rem;font-weight:600;text-decoration:none!important;white-space:nowrap}
.stTabs [data-baseweb="tab-list"]{gap:.15rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{height:2.6rem;padding:0 .7rem;border-radius:10px 10px 0 0;font-weight:500}
.stTabs [aria-selected="true"]{color:var(--wine)!important;font-weight:700}
.stTabs [data-baseweb="tab-highlight"]{background-color:var(--wine);height:3px}
.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:600}
button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,var(--wine-d),var(--wine));border:0;color:#fff}
button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.1);color:#fff}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,div[data-baseweb="textarea"]>div{border-radius:12px!important}
div[data-testid="stExpander"]{border-radius:14px;border:1px solid var(--line)}
div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
</style>"""


CSS_MORE = """<style>
.wh-detail{background:var(--soft);border:1px solid var(--line);border-radius:18px;padding:1rem 1.05rem;margin:.2rem 0 .8rem}
.wh-detail .wh-name{font-size:1.45rem}
.wh-sec{font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;opacity:.65;margin:.9rem 0 .3rem}
.wh-line{display:flex;flex-wrap:wrap;align-items:center;gap:.3rem .5rem;margin:.25rem 0}
.wh-line b{font-family:'Playfair Display',Georgia,serif;font-weight:600}
.wh-sites{display:flex;flex-wrap:wrap;gap:.45rem}
.wh-sites .chip{margin:0;font-size:.85rem;padding:.35rem .85rem}
.wh-notes{font-size:.9rem;opacity:.85;white-space:pre-wrap}
div[data-testid="stRadio"] [role="radiogroup"]{gap:.5rem}
</style>"""
CSS_PHONE = """<style>
html{-webkit-text-size-adjust:100%}
.block-container{padding-top:3.2rem!important;padding-left:.7rem!important;padding-right:.7rem!important;padding-bottom:5rem!important}
input,textarea,[data-baseweb="select"] input,[data-baseweb="select"] div{font-size:16px!important}
.stButton>button,.stDownloadButton>button{min-height:3rem;font-size:1rem;border-radius:14px}
.stTabs [data-baseweb="tab"]{height:3rem;font-size:.95rem;padding:0 .55rem}
.wh-hero{padding:.8rem .95rem;border-radius:16px;margin-bottom:.6rem}
.wh-hero h1{font-size:1.5rem!important}
.wh-hero p{margin:.15rem 0 0;font-size:.85rem}
.wh-grid{grid-template-columns:1fr}
.wh-name{font-size:1.2rem}.wh-zone{font-size:.88rem}
.chip{font-size:.8rem;padding:.12rem .7rem}
.wh-btn{padding:.5rem 1.1rem;font-size:.95rem}
.wh-sitehead h2{font-size:1.35rem!important}
[class*="st-key-prow"] button{justify-content:flex-start;text-align:left;height:auto;min-height:3.4rem;padding:.65rem .9rem;border-radius:14px;white-space:normal;line-height:1.3;font-weight:500}
[class*="st-key-prow"] button p{text-align:left}
div[data-testid="stExpander"] summary{padding:.75rem .9rem}
</style>"""
CSS_PC = """<style>
.block-container{max-width:1240px!important;padding-top:3.5rem!important}
.wh-grid{grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
.wh-hero{padding:1.3rem 1.6rem}
.wh-hero h1{font-size:2.3rem!important}
.wh-hero p{font-size:1rem}
</style>"""


def device_default():
    """Telefono o computer in base al dispositivo che sta aprendo la pagina (si può cambiare con l'interruttore)."""
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        ua = ""
    return "pc" if ua and not re.search(r"Mobile|iPhone|iPod|Android", ua) else "tel"


def hero_html(phone, n_prod, n_shops, n_links):
    if phone:
        return f'<div class="wh-hero"><h1>🍷 Cantina</h1><p>{n_prod} produttori · {n_shops} siti · {n_links} collegamenti</p></div>'
    return ('<div class="wh-hero"><h1>🍷 Cantina</h1><p>Produttori e siti dove sei certo di trovare i loro vini</p><div class="wh-kpi">'
            + "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in (("Produttori", n_prod), ("Siti", n_shops), ("Collegamenti", n_links)))
            + "</div></div>")


def shop_pill(name, shop, text):
    url = (shop or {}).get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        href = url.replace("{q}", quote_plus(text))
        return f'<a class="wh-btn" href="{escape(href, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(name)} ↗</a>'
    return f'<span class="chip">{escape(name)}</span>'


def detail_html(p, shop_map):
    """Scheda di sola lettura del produttore: zona, vini (nome, tipo, uva), siti con link e note."""
    lines = "".join(
        '<div class="wh-line">' + (f"<b>{escape(w)}</b>" if w else "")
        + (f'<span class="chip t-{type_class(t)}">{escape(t)}{" · " + escape(g) if t and g else escape(g)}</span>' if t or g else "") + "</div>"
        for t, g, w in p["styles"] if t or g or w)
    sites = "".join(shop_pill(x, shop_map.get(x), p["name"]) for x in p["sites"])
    return (f'<div class="wh-detail"><div class="wh-name">{escape(p["name"])}</div><div class="wh-zone">{escape(flagged_zone(p) or "Zona non indicata")}</div>'
            + (f'<div class="wh-sec">Vini</div>{lines}' if lines else "")
            + (f'<div class="wh-sec">Dove trovarli</div><div class="wh-sites">{sites}</div>' if sites else "")
            + (f'<div class="wh-sec">Note</div><div class="wh-notes">{escape(p["notes"])}</div>' if p["notes"] else "") + "</div>")


def save_shop(sid, name, country, url):
    with db() as c:
        if sid is None:
            c.execute("INSERT INTO shops(name,country,url) VALUES(?,?,?)", (name, country, url))
        else:
            c.execute("UPDATE shops SET name=?, country=?, url=? WHERE id=?", (name, country, url, sid))
    autosave()


def delete_shop(sid):
    with db() as c:
        c.execute("DELETE FROM shops WHERE id=?", (int(sid),))
    autosave()


def go(sel=None):
    """Apre la scheda di un produttore (id), il modulo «nuovo» ('new') oppure torna all'elenco (None)."""
    ss = st.session_state
    ss.sel, ss.editing, ss.confirm, ss.last_tbl, ss.tv = sel, None, None, None, ss.tv + 1


def pick(box, label, options, current, key):
    """Menu a tendina con i valori già usati (evita errori di battitura) e la voce «➕ Altro…» per i valori nuovi."""
    opts = sorted({o for o in options if o} | ({current} if current else set()), key=str.lower)
    choice = box.selectbox(label, opts + ["➕ Altro…"], index=opts.index(current) if current in opts else None,
                           placeholder="Scegli…", key=key)
    return box.text_input(f"{label}: nuovo valore", key=key + "n").strip() if choice == "➕ Altro…" else (choice or "")


def producer_form(pid, allp, shops):
    cur = next((p for p in allp if p["id"] == pid), None) if pid else None
    key_id = pid or "new"
    if st.session_state.get("ov_owner") != key_id:  # cambiando produttore, le proposte non salvate vengono dimenticate
        st.session_state.ov, st.session_state.sug, st.session_state.ov_owner = None, None, key_id
    base = st.session_state.ov or (cur or {"name": "", "country": "", "region": "", "subregion": "", "notes": "", "styles": [], "sites": []})
    k = f"pf{key_id}_{st.session_state.fv}"
    has_key = bool(secret("GEMINI_API_KEY") or secret("ANTHROPIC_API_KEY"))
    name = st.text_input("Produttore", base["name"], key=k + "n")
    ai_click = st.button("✨ Compila automaticamente", key=k + "ai", disabled=not (name.strip() and has_key),
                         help="Propone paese, regione, provincia, tipi di vino e uve: controlla sempre prima di salvare")
    if not has_key:
        st.caption("Per la compilazione automatica serve una chiave GEMINI_API_KEY nei Secrets.")
    overwrite = st.checkbox("Sostituisci anche i campi già compilati", key=k + "ow") if cur else False
    c1, c2 = st.columns(2)
    country = pick(c1, "Paese", [p["country"] for p in allp] + KNOWN_COUNTRIES, base["country"], k + "c")
    region = pick(c2, "Regione", [p["region"] for p in allp], base["region"], k + "r")
    subregion = pick(st, "Provincia / sotto-regione", [p["subregion"] for p in allp], base["subregion"], k + "p")
    types = sorted({t for p in allp for t, _, _ in p["styles"] if t} | set(BASE_TYPES))
    st.markdown("**Vini**")
    wdf = pd.DataFrame([(w, t, g) for t, g, w in base["styles"]], columns=["Vino", "Tipo", "Uva"])
    wdf = st.data_editor(wdf, num_rows="dynamic", hide_index=True, use_container_width=True, key=k + "s",
                         column_config={"Vino": "Vino", "Tipo": st.column_config.SelectboxColumn("Tipo", options=types), "Uva": "Uva"})
    sites = st.multiselect("🛒 Siti dove hai verificato che li trovi", [s["name"] for s in shops], default=base["sites"], key=k + "w")
    notes = st.text_area("Note", base["notes"], key=k + "t", height=80)
    styles = [(_cl(t), _cl(g), _cl(w)) for w, t, g in wdf[["Vino", "Tipo", "Uva"]].itertuples(index=False)]
    if ai_click:
        opts = {"countries": sorted({p["country"] for p in allp if p["country"]} | set(KNOWN_COUNTRIES)),
                "regions": sorted({p["region"] for p in allp if p["region"]}), "subs": sorted({p["subregion"] for p in allp if p["subregion"]}),
                "types": types, "grapes": sorted({g for p in allp for _, g, _ in p["styles"] if g})}
        res = None
        with st.spinner("Ricerca in corso… può richiedere fino a un minuto"):
            try:
                res = autofill(name.strip(), opts)
            except Exception as e:
                st.error(f"Errore: {e}")
        if res is not None:
            if res.get("error"):
                st.warning("Produttore non riconosciuto: compila a mano.")
            else:
                sug = clean_suggestion(res, opts)
                snap = {"name": name.strip(), "country": country, "region": region, "subregion": subregion, "notes": notes,
                        "styles": styles, "sites": sites}
                merged, sug["note"] = merge_suggestion(snap, sug, overwrite)
                st.session_state.ov, st.session_state.sug = merged, sug
                st.session_state.fv += 1
                st.rerun()
    sug = st.session_state.sug
    if sug and sug.get("note"):
        st.warning(sug["note"])
    if sug:
        with st.expander("🤖 Proposta ricevuta: da controllare", expanded=False):
            st.write(f"**Zona:** {' · '.join(x for x in (sug['country'], sug['region'], sug['subregion']) if x) or '—'}")
            st.write("**Tipi e uve:** " + (", ".join(f"{t} {g}".strip() for t, g in sug["styles"]) or "—"))
            if sug["sources"]:
                st.markdown("**Fonti consultate sul web:**\n" + "\n".join(
                    "- [" + (x["title"] or x["uri"]).replace("[", "(").replace("]", ")") + "](" + x["uri"] + ")" for x in sug["sources"][:10]))
            else:
                st.caption("Nessuna fonte web: la proposta viene dalla memoria del modello e può contenere errori.")
    ex = find_producer(name, allp) if not pid and name.strip() else None
    if ex:
        st.info(f"📎 «{ex['name']}» è già in rubrica: salvando verranno aggiunti solo vini e siti nuovi, senza togliere nulla.")
    if st.button(f"🔀 Unisci a «{ex['name']}»" if ex else "💾 Salva", key=k + "b", type="primary"):
        if not name.strip():
            st.error("Inserisci il nome del produttore.")
        else:
            try:
                saved = save_producer(pid, name.strip(), country, region, subregion, notes.strip(), styles, sites, merge_into=ex["id"] if ex else None)
                autosave()
                st.session_state.ov, st.session_state.sug = None, None
                st.session_state.fv += 1
                go(saved)
                st.toast("Salvato ✅")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("Esiste già un produttore con questo nome.")


# ───────────────────────── Interfaccia ─────────────────────────
st.set_page_config(page_title="Cantina", page_icon="🍷", layout="wide")
init()
first_boot()
for _k, _v in (("fv", 0), ("tv", 0), ("confirm", None), ("ov", None), ("sug", None), ("ov_owner", None), ("sel", None), ("editing", None),
               ("last_tbl", None), ("last_txt", ""), ("limit", 40), ("confirm_shop", None)):
    st.session_state.setdefault(_k, _v)
_opts = ["📱 Telefono", "💻 Computer"]
if "vista" not in st.session_state:
    st.session_state.vista = _opts[1] if (st.query_params.get("v") or device_default()) == "pc" else _opts[0]
st.radio("Vista", _opts, horizontal=True, key="vista", label_visibility="collapsed")
PHONE = st.session_state.vista == _opts[0]
st.query_params["v"] = "tel" if PHONE else "pc"
st.markdown(CSS + CSS_MORE + (CSS_PHONE if PHONE else CSS_PC), unsafe_allow_html=True)
_n = lambda sql: int(q(sql).iloc[0, 0])
st.markdown(hero_html(PHONE, _n("SELECT COUNT(*) FROM producers"), _n("SELECT COUNT(*) FROM shops"), _n("SELECT COUNT(*) FROM avail")),
            unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["🛒 Per sito", "📒 Produttori", "🌐 Siti", "💾 Dati"])
allp = load_all()
shops_df = q("SELECT id,name,country,url FROM shops ORDER BY name COLLATE NOCASE")
shops = shops_df.to_dict("records")

# ── Per sito
with t1:
    if not allp:
        st.info("Nessun produttore: importa il tuo foglio dalla scheda 💾 Dati o aggiungili dalla scheda 📒 Produttori.")
    else:
        NONE = "— senza sito assegnato —"
        counts = {s["name"]: sum(s["name"] in p["sites"] for p in allp) for s in shops}
        options = sorted(counts, key=lambda n: (-counts[n], n.lower())) + [NONE]
        counts[NONE] = sum(not p["sites"] for p in allp)
        sel = st.selectbox("🛒 Dove vuoi comprare?", options, format_func=lambda n: f"{n} ({counts[n]})")
        shop = next((s for s in shops if s["name"] == sel), None)
        items = [p for p in allp if (not p["sites"] if sel == NONE else sel in p["sites"])]
        with st.expander("🎛️ Filtri"):
            txt = st.text_input("Cerca produttore")
            f_c = st.multiselect("Paese", sorted({p["country"] for p in items if p["country"]}))
            f_r = st.multiselect("Regione", sorted({p["region"] for p in items if p["region"]}))
            f_s = st.multiselect("Provincia / sotto-regione", sorted({p["subregion"] for p in items if p["subregion"]}))
            f_t = st.multiselect("Tipo", sorted({t for p in items for t, _, _ in p["styles"] if t}))
            f_u = st.multiselect("Uva", sorted({g for p in items for _, g, _ in p["styles"] if g}))
        items = [p for p in items if (not txt or norm(txt) in norm(p["name"])) and (not f_c or p["country"] in f_c)
                 and (not f_r or p["region"] in f_r) and (not f_s or p["subregion"] in f_s)
                 and (not f_t or any(t in f_t for t, _, _ in p["styles"])) and (not f_u or any(g in f_u for _, g, _ in p["styles"]))]
        home = shop_home(shop)
        st.markdown(f'<div class="wh-sitehead"><h2>{escape("Senza sito assegnato" if sel == NONE else sel)}</h2>'
                    f'<span>{len(items)} produttori</span>' + (f'<a href="{escape(home, quote=True)}" target="_blank" rel="noopener noreferrer">Apri sito ↗</a>' if home else "")
                    + "</div>", unsafe_allow_html=True)
        if items:
            st.markdown(cards_html(items, shop), unsafe_allow_html=True)
        else:
            st.info("Nessun produttore con questi filtri.")

# ── Produttori
def filter_view(txt):
    words = norm(txt).split()
    return [p for p in allp if all(w in norm(" ".join([p["name"], p["country"], p["region"], p["subregion"], *p["sites"],
                                                       *(f"{w2} {t} {g}" for t, g, w2 in p["styles"])])) for w in words)]


def producer_detail(sel):
    if sel == "new":
        st.markdown("### ➕ Nuovo produttore")
        producer_form(None, allp, shops)
        return
    cur = next((p for p in allp if p["id"] == sel), None)
    if cur is None:
        st.info("Questo produttore non esiste più.")
        return
    st.markdown(detail_html(cur, {x["name"]: x for x in shops}), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("✏️ Modifica", key=f"ed{sel}", use_container_width=True):
        st.session_state.editing, st.session_state.confirm = sel, None
    if c2.button("🗑️ Elimina", key=f"rm{sel}", use_container_width=True):
        st.session_state.confirm, st.session_state.editing = sel, None
    if st.session_state.confirm == sel:
        st.warning(f"Eliminare «{cur['name']}»? L'operazione non si può annullare.")
        y, n = st.columns(2)
        if y.button("Sì, elimina", key=f"yes{sel}", type="primary", use_container_width=True):
            delete_producer(sel)
            go(None)
            st.toast("Produttore eliminato 🗑️")
            st.rerun()
        if n.button("Annulla", key=f"no{sel}", use_container_width=True):
            st.session_state.confirm = None
            st.rerun()
    if st.session_state.editing == sel:
        producer_form(sel, allp, shops)
        if st.button("✖ Chiudi modifica", key=f"cl{sel}"):
            st.session_state.editing = None
            st.rerun()


def producers_list_phone():
    if st.button("➕ Nuovo produttore", type="primary", use_container_width=True):
        go("new")
        st.rerun()
    txt = st.text_input("Cerca", key="ps", placeholder="🔎 Produttore, regione, uva, sito…", label_visibility="collapsed")
    if txt != st.session_state.last_txt:
        st.session_state.last_txt, st.session_state.limit = txt, 40
    view = filter_view(txt)
    st.caption(f"{len(view)} produttori")
    for p in view[:st.session_state.limit]:
        label = f"{FLAGS.get(p['country'], '🍷')}  {p['name']}" + (f"  ·  {p['region']}" if p["region"] else "")
        if st.button(label, key=f"prow{p['id']}", use_container_width=True):
            go(p["id"])
            st.rerun()
    if len(view) > st.session_state.limit and st.button(f"Mostra altri ({len(view) - st.session_state.limit})", key="more", use_container_width=True):
        st.session_state.limit += 40
        st.rerun()


def producers_list_pc():
    if st.button("➕ Nuovo produttore", type="primary"):
        go("new")
        st.rerun()
    txt = st.text_input("🔎 Cerca per produttore, regione, uva o sito", key="ps")
    view = filter_view(txt)
    st.caption(f"{len(view)} produttori · clicca una riga per aprirla")
    if view:
        table = pd.DataFrame({"Produttore": [p["name"] for p in view], "Zona": [flagged_zone(p) for p in view],
                              "Siti": [", ".join(p["sites"]) for p in view]})
        ev = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                          height=560, key=f"ptbl_{st.session_state.tv}_{txt.strip()}")
        if ev.selection.rows:
            pid = view[ev.selection.rows[0]]["id"]
            if pid != st.session_state.last_tbl:
                st.session_state.last_tbl, st.session_state.sel, st.session_state.editing, st.session_state.confirm = pid, pid, None, None


with t2:
    if not allp and st.session_state.sel != "new":
        st.info("Rubrica vuota: aggiungi il primo produttore o importa il tuo foglio dalla scheda 💾 Dati.")
        if st.button("➕ Aggiungi il primo produttore", type="primary"):
            go("new")
            st.rerun()
    elif PHONE:
        if st.session_state.sel is None:
            producers_list_phone()
        else:
            if st.button("← Elenco", key="back"):
                go(None)
                st.rerun()
            producer_detail(st.session_state.sel)
    else:
        left, right = st.columns([2, 3], gap="large")
        with left:
            producers_list_pc()
        with right:
            if st.session_state.sel is None:
                st.info("Seleziona un produttore dall'elenco per vederne i vini e modificarlo.")
            else:
                producer_detail(st.session_state.sel)

# ── Siti
def shop_fields(sid, s0, prefix):
    nm = st.text_input("Nome", s0["name"], key=f"{prefix}n{sid}")
    cs = [""] + sorted(set(KNOWN_COUNTRIES) | ({s0["country"]} if s0["country"] else set()))
    ct = st.selectbox("Paese", cs, index=cs.index(s0["country"]) if s0["country"] in cs else 0, key=f"{prefix}c{sid}")
    url = st.text_input("Indirizzo di ricerca (con {q} al posto del testo cercato)", s0["url"], key=f"{prefix}u{sid}", placeholder="https://sito.it/cerca?q={q}")
    return _cl(nm), ct, _cl(url)


def shops_phone():
    st.caption("Tocca un sito per modificarlo.")
    for s0 in shops:
        s0 = {**s0, "country": s0["country"] or "", "url": s0["url"] or ""}
        n_p = sum(s0["name"] in p["sites"] for p in allp)
        with st.expander(f"{s0['name']} · {n_p} produttori"):
            nm, ct, url = shop_fields(s0["id"], s0, "sh")
            if st.button("💾 Salva", key=f"shs{s0['id']}", type="primary", use_container_width=True):
                if not nm:
                    st.error("Il nome non può essere vuoto.")
                else:
                    try:
                        save_shop(s0["id"], nm, ct, url)
                        st.toast("Sito salvato ✅")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Esiste già un sito con questo nome.")
            if st.button("🗑️ Elimina sito", key=f"shd{s0['id']}", use_container_width=True):
                st.session_state.confirm_shop = s0["id"]
            if st.session_state.confirm_shop == s0["id"]:
                st.warning(f"Eliminare «{s0['name']}»? Verranno tolti anche i suoi collegamenti ai produttori.")
                if st.button("Sì, elimina", key=f"shy{s0['id']}", type="primary", use_container_width=True):
                    delete_shop(s0["id"])
                    st.session_state.confirm_shop = None
                    st.rerun()
                if st.button("Annulla", key=f"shno{s0['id']}", use_container_width=True):
                    st.session_state.confirm_shop = None
                    st.rerun()
    with st.expander("➕ Nuovo sito"):
        nm, ct, url = shop_fields("new", {"name": "", "country": "", "url": ""}, "sh")
        if st.button("Aggiungi", key="shadd", type="primary", use_container_width=True):
            if not nm:
                st.error("Scrivi il nome del sito.")
            else:
                try:
                    save_shop(None, nm, ct, url)
                    st.toast("Sito aggiunto ✅")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Esiste già un sito con questo nome.")


with t3:
    if PHONE:
        shops_phone()
    else:
        st.caption("Per ogni sito: nome, paese e indirizzo di ricerca, con {q} al posto del testo cercato (es. https://sito.it/cerca?q={q}). "
                   "Serve per il pulsante «Cerca» sulle schede. Eliminare un sito toglie anche i suoi collegamenti ai produttori.")
        ed = st.data_editor(shops_df.assign(country=shops_df["country"].fillna(""), url=shops_df["url"].fillna("")), num_rows="dynamic", hide_index=True,
                            use_container_width=True, key="shops_ed",
                            column_config={"id": None, "name": "Sito", "country": st.column_config.SelectboxColumn("Paese", options=KNOWN_COUNTRIES),
                                           "url": "Indirizzo di ricerca"})
        if st.button("💾 Salva siti", type="primary"):
            keep = [int(i) for i in ed["id"].dropna()] or [-1]
            with db() as c:
                c.execute(f"DELETE FROM shops WHERE id NOT IN ({','.join('?' * len(keep))})", keep)
                for _, r in ed.iterrows():
                    nm = _cl(r["name"])
                    if not nm:
                        continue
                    vals = (nm, _cl(r["country"]), _cl(r["url"]))
                    if pd.notna(r["id"]):
                        c.execute("UPDATE shops SET name=?, country=?, url=? WHERE id=?", (*vals, int(r["id"])))
                    else:
                        c.execute("INSERT OR IGNORE INTO shops(name,country,url) VALUES(?,?,?)", vals)
            autosave()
            st.toast("Siti salvati ✅")
            st.rerun()


# ── Dati
with t4:
    if gh_conf():
        st.success("☁️ Salvataggio automatico su GitHub attivo: i dati sopravvivono ai riavvii dell'app.")
    else:
        st.warning("Salvataggio automatico non configurato: i dati si azzerano a ogni riavvio. Aggiungi GITHUB_TOKEN e GITHUB_REPO nei Secrets.")
    st.subheader("📥 Importa il tuo foglio Excel")
    st.caption("Carica il file con le colonne Paese, Regione, Produttore, Tipo 1/Uva 1…, Sito 1…. I produttori già presenti vengono uniti senza togliere nulla.")
    up = st.file_uploader("File .xlsx", type=["xlsx"], key="xl")
    if up:
        try:
            rows, new_shops, fixes = read_old_xlsx(up.getvalue())
        except Exception as e:
            rows, new_shops, fixes = None, None, [f"Non riesco a leggere il file: {e}"]
        if rows is None:
            st.error(fixes[0])
        else:
            known = {pkey(p["name"]) for p in allp}
            st.caption(f"{len(rows)} produttori ({sum(pkey(r['name']) not in known for r in rows)} nuovi) · {len(new_shops)} siti")
            if fixes:
                st.info("Correzioni automatiche: " + "; ".join(fixes))
            st.dataframe(pd.DataFrame([{"Produttore": r["name"], "Paese": r["country"], "Regione": r["region"], "Sotto-regione": r["subregion"],
                                        "Tipi e uve": styles_text(r), "Siti": ", ".join(r["sites"]),
                                        "Stato": "già presente (unito)" if pkey(r["name"]) in known else "nuovo"} for r in rows]),
                         hide_index=True, use_container_width=True)
            if st.button("📥 Importa", type="primary"):
                a, m = import_rows(rows, new_shops)
                autosave()
                st.session_state.tv += 1
                st.toast(f"Importati: {a} produttori nuovi, {m} uniti ✅")
                st.rerun()
    st.divider()
    st.subheader("📤 Esporta")
    buf = io.BytesIO()
    exp = export_df()
    exp.to_excel(buf, index=False, sheet_name="Produttori")
    st.download_button("⬇️ Scarica Excel", buf.getvalue(), "cantina.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("⬇️ Scarica backup completo (JSON)", json.dumps(dump(), ensure_ascii=False, default=str), "cantina_backup.json", "application/json")
    bk = st.file_uploader("Ripristina da backup JSON", type="json", key="bk")
    if bk and st.button("♻️ Ripristina (sovrascrive tutto)"):
        restore(json.load(bk))
        autosave()
        st.rerun()
    if gh_conf() and st.button("Salva ora su GitHub"):
        st.success("Salvato.") if push_backup() else st.error("Salvataggio non riuscito: controlla GITHUB_TOKEN e GITHUB_REPO.")
