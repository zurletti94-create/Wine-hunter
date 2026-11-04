import json
import re
import sqlite3
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

DB = "wine.db"
UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept-Language": "it,en;q=0.8,fr;q=0.6,de;q=0.6,es;q=0.6",
}
COUNTRIES = ["Italia", "Francia", "Spagna", "Germania", "Austria"]
TYPES = ["rosso", "bianco", "rosato", "spumante", "dolce", "altro"]
STOP = {"cantina", "cantine", "azienda", "agricola", "tenuta", "fattoria", "podere", "castello",
        "marchesi", "domaine", "chateau", "maison", "bodega", "bodegas", "weingut", "winery",
        "estate", "famiglia", "vini", "vigneti", "des", "les", "del", "della", "dei", "von"}

# Template di ricerca di partenza: verificali/correggili nella scheda "Siti" con il pulsante Prova.
DEFAULT_SHOPS = [
    ("Tannico", "Italia", "https://www.tannico.it/catalogsearch/result/?q={q}"),
    ("Callmewine", "Italia", "https://www.callmewine.com/search?q={q}"),
    ("Millésima", "Francia", "https://www.millesima.fr/catalogsearch/result/?q={q}"),
    ("Vinatis", "Francia", "https://www.vinatis.com/recherche?search_query={q}"),
    ("Vinissimus", "Spagna", "https://www.vinissimus.com/es/buscar/?q={q}"),
    ("Bodeboca", "Spagna", "https://www.bodeboca.com/buscar?q={q}"),
    ("Weinfreunde", "Germania", "https://www.weinfreunde.de/search?sSearch={q}"),
    ("Hawesko", "Germania", "https://www.hawesko.de/suche?search={q}"),
    ("Wein & Co", "Austria", "https://www.weinco.at/de/search?q={q}"),
]


# ───────────────────────── Database ─────────────────────────
def db():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys=ON")
    return con


def q(sql, p=()):
    con = db()
    try:
        return pd.read_sql_query(sql, con, params=p)
    finally:
        con.close()


def init():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS producers(id INTEGER PRIMARY KEY, name TEXT UNIQUE, country TEXT, region TEXT, subregion TEXT);
        CREATE TABLE IF NOT EXISTS wines(id INTEGER PRIMARY KEY, producer_id INTEGER REFERENCES producers(id) ON DELETE CASCADE, name TEXT, wine_type TEXT, grapes TEXT);
        CREATE TABLE IF NOT EXISTS shops(id INTEGER PRIMARY KEY, name TEXT UNIQUE, country TEXT, url TEXT, enabled INTEGER DEFAULT 1);
        CREATE TABLE IF NOT EXISTS offers(id INTEGER PRIMARY KEY, wine_id INTEGER REFERENCES wines(id) ON DELETE CASCADE, shop_id INTEGER REFERENCES shops(id) ON DELETE CASCADE, title TEXT, price REAL, currency TEXT, link TEXT, found TEXT);
        """)
        if not c.execute("SELECT 1 FROM shops").fetchone():
            c.executemany("INSERT INTO shops(name,country,url) VALUES(?,?,?)", DEFAULT_SHOPS)


def sync_rows(c, table, parent_col, parent_id, df, cols):
    """Allinea una tabella figlia a un data_editor (id NaN = nuova riga)."""
    keep = [int(i) for i in df["id"].dropna()] or [-1]
    where, args = (f"{parent_col}=? AND " if parent_col else ""), ([parent_id] if parent_col else [])
    c.execute(f"DELETE FROM {table} WHERE {where}id NOT IN ({','.join('?' * len(keep))})", args + keep)
    for _, r in df.iterrows():
        vals = [("" if pd.isna(r[k]) else r[k]) for k in cols]
        if not str(vals[0]).strip():
            continue
        if pd.notna(r["id"]):
            c.execute(f"UPDATE {table} SET {','.join(k + '=?' for k in cols)} WHERE id=?", vals + [int(r["id"])])
        elif parent_col:
            c.execute(f"INSERT INTO {table}({parent_col},{','.join(cols)}) VALUES(?,{','.join('?' * len(cols))})",
                      [parent_id] + vals)
        else:
            c.execute(f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)


def save_producer(pid, name, country, region, sub, wines):
    with db() as c:
        if pid is None:
            pid = c.execute("INSERT INTO producers(name,country,region,subregion) VALUES(?,?,?,?)",
                            (name, country, region, sub)).lastrowid
        else:
            c.execute("UPDATE producers SET name=?,country=?,region=?,subregion=? WHERE id=?",
                      (name, country, region, sub, pid))
        sync_rows(c, "wines", "producer_id", pid, wines, ["name", "wine_type", "grapes"])


# ───────────────────────── Compilazione automatica (LLM) ─────────────────────────
def secret(k, d=None):
    try:
        return st.secrets.get(k, d)
    except Exception:
        return d


def autofill(producer):
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    prompt = f"""Sei un sommelier esperto. Per il produttore di vino "{producer}" rispondi SOLO con JSON valido, senza altro testo:
{{"name":"nome ufficiale","country":"Italia|Francia|Spagna|Germania|Austria|altro","region":"","subregion":"denominazione/sottozona principale","wines":[{{"name":"","wine_type":"rosso|bianco|rosato|spumante|dolce|altro","grapes":"uve separate da virgola"}}]}}
Elenca i vini principali realmente prodotti dalla cantina (max 15). Non inventare: se non conosci il produttore rispondi {{"error":"sconosciuto"}}."""
    r = client.messages.create(model=secret("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=1500,
                               messages=[{"role": "user", "content": prompt}])
    return json.loads(re.search(r"\{.*\}", r.content[0].text, re.S).group(0))


# ───────────────────────── Ricerca offerte ─────────────────────────
PRICE = re.compile(r"(?:€|EUR)\s?(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)|(\d{1,4}(?:[.,]\d{1,2})?)\s?(?:€|EUR)")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", unicodedata.normalize("NFKD", str(s).lower()).encode("ascii", "ignore").decode())


def tokens(s):
    return [t for t in norm(s).split() if len(t) > 2 and t not in STOP]


def to_float(s):
    s = str(s).strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def fetch(url):
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        return r.text, None
    except Exception as e:
        return None, str(e)[:100]


def parse(html, base):
    """Estrae (titolo, prezzo, valuta, link): prima da JSON-LD schema.org, poi euristica sui link."""
    soup, out = BeautifulSoup(html, "html.parser"), []

    def walk(o):
        if isinstance(o, list):
            for i in o:
                walk(i)
        elif isinstance(o, dict):
            t = o.get("@type")
            if t == "Product" or (isinstance(t, list) and "Product" in t):
                off = o.get("offers") or {}
                off = (off[0] if off else {}) if isinstance(off, list) else off
                p = off.get("price") or off.get("lowPrice")
                if p and o.get("name"):
                    out.append((o["name"], to_float(p), off.get("priceCurrency", "EUR"),
                                urljoin(base, o.get("url") or off.get("url") or base)))
            for v in o.values():
                if isinstance(v, (dict, list)):
                    walk(v)

    for s in soup.find_all("script", type="application/ld+json"):
        try:
            walk(json.loads(s.string or ""))
        except Exception:
            pass
    if out:
        return out
    for a in soup.find_all("a", href=True):
        title = a.get_text(" ", strip=True)
        if len(title) < 8:
            continue
        box, m = a, None
        for _ in range(4):
            if box is None:
                break
            txt = box.get_text(" ", strip=True)
            if len(txt) > 600:
                break
            m = PRICE.search(txt)
            if m:
                break
            box = box.parent
        if m:
            out.append((title, to_float(m.group(1) or m.group(2)), "EUR", urljoin(base, a["href"])))
    return out


def search_one(w, s):
    html, err = fetch(s["url"].replace("{q}", quote_plus(f'{w["producer"]} {w["wine"]}')))
    if err:
        return w["wine_id"], s["id"], [], f'{s["name"]}: {err}'
    ptok, wtok, res, seen = tokens(w["producer"]), tokens(w["wine"]), [], set()
    for title, price, cur, link in parse(html, s["url"]):
        t = norm(title)
        if price is None or not 3 <= price <= 20000 or link in seen:
            continue
        if (ptok and not any(x in t for x in ptok)) or (wtok and not any(x in t for x in wtok)):
            continue
        seen.add(link)
        res.append((title[:200], price, cur, link))
    return w["wine_id"], s["id"], res, None


def save_offers(wid, sid, res):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with db() as c:
        c.execute("DELETE FROM offers WHERE wine_id=? AND shop_id=?", (wid, sid))
        c.executemany("INSERT INTO offers(wine_id,shop_id,title,price,currency,link,found) VALUES(?,?,?,?,?,?,?)",
                      [(wid, sid, t, p, cu, l, now) for t, p, cu, l in res])


# ───────────────────────── UI ─────────────────────────
def producer_form(pid, d, wines, key):
    name = st.text_input("Produttore", d["name"], key=f"n{key}")
    c1, c2, c3 = st.columns(3)
    country = c1.text_input("Paese", d["country"], key=f"c{key}")
    region = c2.text_input("Regione", d["region"], key=f"r{key}")
    sub = c3.text_input("Sotto-regione", d["subregion"], key=f"s{key}")
    wines = st.data_editor(
        wines, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"w{key}",
        column_config={"id": None, "name": "Vino",
                       "wine_type": st.column_config.SelectboxColumn("Tipo", options=TYPES),
                       "grapes": "Uve"})
    b1, b2 = st.columns(2)
    if b1.button("💾 Salva", key=f"save{key}", type="primary"):
        if not name.strip():
            st.error("Inserisci il nome del produttore.")
        else:
            try:
                save_producer(pid, name.strip(), country, region, sub, wines)
                st.session_state.draft = None
                st.toast("Salvato ✅")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("Produttore già presente.")
    if pid and b2.button("🗑️ Elimina", key=f"del{key}"):
        with db() as c:
            c.execute("DELETE FROM producers WHERE id=?", (int(pid),))
        st.rerun()


st.set_page_config(page_title="Wine Hunter", page_icon="🍷", layout="wide")
init()
st.title("🍷 Wine Hunter")
t1, t2, t3, t4, t5 = st.tabs(["📒 Rubrica", "➕ Nuovo", "🛒 Offerte", "🌐 Siti", "💾 Backup"])

# ── Rubrica
with t1:
    prod = q("SELECT * FROM producers ORDER BY name")
    if prod.empty:
        st.info("Rubrica vuota: aggiungi un produttore dalla scheda ➕ Nuovo.")
    else:
        full = q("""SELECT p.name Produttore, p.country Paese, p.region Regione, p.subregion Sottoregione,
                    w.name Vino, w.wine_type Tipo, w.grapes Uve
                    FROM producers p LEFT JOIN wines w ON w.producer_id=p.id ORDER BY p.name, w.name""")
        txt = st.text_input("🔎 Cerca")
        if txt:
            full = full[full.apply(lambda r: txt.lower() in " ".join(map(str, r)).lower(), axis=1)]
        st.dataframe(full, use_container_width=True, hide_index=True)
        st.divider()
        sel = st.selectbox("✏️ Modifica produttore", prod["name"])
        row = prod[prod["name"] == sel].iloc[0]
        wdf = q("SELECT id,name,wine_type,grapes FROM wines WHERE producer_id=? ORDER BY name", (int(row["id"]),))
        producer_form(int(row["id"]), {k: row[k] or "" for k in ["name", "country", "region", "subregion"]},
                      wdf, f"e{row['id']}")

# ── Nuovo produttore
with t2:
    st.session_state.setdefault("dv", 0)
    name = st.text_input("Nome del produttore", key="newname")
    if st.button("✨ Compila automaticamente", disabled=not name.strip()):
        draft = {"name": name.strip(), "country": "", "region": "", "subregion": "", "wines": []}
        if not secret("ANTHROPIC_API_KEY"):
            st.warning("Nessuna ANTHROPIC_API_KEY nei secrets: compila i campi a mano.")
        else:
            with st.spinner("Ricerca informazioni…"):
                try:
                    res = autofill(name)
                    if res.get("error"):
                        st.warning("Produttore non riconosciuto: compila a mano.")
                    else:
                        draft.update({k: res.get(k, "") for k in ["name", "country", "region", "subregion"]})
                        draft["wines"] = res.get("wines", [])
                except Exception as e:
                    st.error(f"Errore: {e}")
        st.session_state.draft = draft
        st.session_state.dv += 1
    d = st.session_state.get("draft")
    if d:
        st.caption("⚠️ Dati generati da un modello AI: verificali prima di salvare.")
        wdf = pd.DataFrame(d["wines"], columns=["name", "wine_type", "grapes"])
        wdf["wine_type"] = wdf["wine_type"].where(wdf["wine_type"].isin(TYPES), "altro")
        wdf.insert(0, "id", pd.Series([None] * len(wdf), dtype="float"))
        producer_form(None, d, wdf, f"new{st.session_state.dv}")

# ── Offerte
with t3:
    wines = q("SELECT w.id wine_id, w.name wine, p.name producer FROM wines w JOIN producers p ON p.id=w.producer_id")
    shops = q("SELECT * FROM shops WHERE enabled=1")
    if wines.empty:
        st.info("Aggiungi prima produttori e vini.")
    else:
        with st.expander("🔎 Avvia ricerca", expanded=True):
            sp = st.multiselect("Produttori (vuoto = tutti)", sorted(wines["producer"].unique()))
            sc = st.multiselect("Paesi dei siti", COUNTRIES, default=COUNTRIES)
            wl = wines[wines["producer"].isin(sp)] if sp else wines
            sl = shops[shops["country"].isin(sc)]
            jobs = [(w, s) for w in wl.to_dict("records") for s in sl.to_dict("records")]
            st.caption(f"{len(jobs)} ricerche ({len(wl)} vini × {len(sl)} siti)")
            if st.button("Cerca offerte", type="primary", disabled=not jobs):
                bar, errs = st.progress(0.0), []
                with ThreadPoolExecutor(8) as ex:
                    for i, (wid, sid, res, err) in enumerate(ex.map(lambda j: search_one(*j), jobs), 1):
                        save_offers(wid, sid, res)
                        if err:
                            errs.append(err)
                        bar.progress(i / len(jobs))
                st.success("Ricerca completata.")
                if errs:
                    with st.expander(f"{len(errs)} errori"):
                        st.write("\n\n".join(sorted(set(errs))[:30]))
        off = q("""SELECT p.name Produttore, w.name Vino, w.wine_type Tipo, s.name Sito, s.country Paese,
                   o.title Titolo, o.price Prezzo, o.link Link, o.found Aggiornato
                   FROM offers o JOIN wines w ON w.id=o.wine_id JOIN producers p ON p.id=w.producer_id
                   JOIN shops s ON s.id=o.shop_id""")
        if off.empty:
            st.info("Nessuna offerta salvata: avvia una ricerca.")
        else:
            f1, f2 = st.columns(2)
            fp = f1.multiselect("Paese del sito", COUNTRIES)
            fs = f2.multiselect("Sito", sorted(off["Sito"].unique()))
            fprod = st.multiselect("Produttore", sorted(off["Produttore"].unique()))
            lo, hi = float(off["Prezzo"].min()), float(off["Prezzo"].max())
            pr = st.slider("Prezzo (€)", lo, hi, (lo, hi)) if lo < hi else (lo, hi)
            if fp:
                off = off[off["Paese"].isin(fp)]
            if fs:
                off = off[off["Sito"].isin(fs)]
            if fprod:
                off = off[off["Produttore"].isin(fprod)]
            off = off[off["Prezzo"].between(*pr)].sort_values("Prezzo")
            st.caption(f"{len(off)} bottiglie")
            st.dataframe(off, use_container_width=True, hide_index=True, column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="Apri"),
                "Prezzo": st.column_config.NumberColumn("Prezzo", format="€ %.2f")})

# ── Siti
with t4:
    st.caption("Usa {q} nell'URL al posto del testo cercato. Aggiungi righe per nuovi siti.")
    sh = q("SELECT id,name,country,url,enabled FROM shops ORDER BY country,name")
    sh["enabled"] = sh["enabled"].astype(bool)
    ed = st.data_editor(sh, num_rows="dynamic", hide_index=True, use_container_width=True, key="shops",
                        column_config={"id": None, "name": "Nome", "url": "URL di ricerca", "enabled": "Attivo",
                                       "country": st.column_config.SelectboxColumn("Paese", options=COUNTRIES)})
    if st.button("💾 Salva siti"):
        ed["enabled"] = ed["enabled"].fillna(True).astype(int)
        with db() as c:
            sync_rows(c, "shops", None, None, ed, ["name", "country", "url", "enabled"])
        st.toast("Siti salvati ✅")
        st.rerun()
    st.subheader("🧪 Prova un sito")
    if not sh.empty:
        tsite = st.selectbox("Sito", sh["name"])
        tq = st.text_input("Ricerca di prova", "Sassicaia")
        if st.button("Prova"):
            url = sh[sh["name"] == tsite].iloc[0]["url"].replace("{q}", quote_plus(tq))
            html, err = fetch(url)
            if err:
                st.error(err)
            else:
                items = parse(html, url)
                st.write(f"{len(items)} prodotti trovati")
                st.dataframe(pd.DataFrame(items, columns=["Titolo", "Prezzo", "Valuta", "Link"]), hide_index=True)
                if not items:
                    st.info("Nessun risultato: il sito potrebbe caricare i prodotti via JavaScript o bloccare i bot.")

# ── Backup
with t5:
    st.warning("Su Streamlit Community Cloud il disco si azzera a ogni riavvio dell'app: scarica un backup regolarmente.")
    TABLES = ["producers", "wines", "shops", "offers"]
    data = {t: q(f"SELECT * FROM {t}").to_dict("records") for t in TABLES}
    st.download_button("⬇️ Scarica backup", json.dumps(data, ensure_ascii=False, default=str),
                       "wine_backup.json", "application/json")
    up = st.file_uploader("Ripristina da backup", type="json")
    if up and st.button("♻️ Ripristina (sovrascrive tutto)"):
        data = json.load(up)
        with db() as c:
            for t in reversed(TABLES):
                c.execute(f"DELETE FROM {t}")
            for t in TABLES:
                df = pd.DataFrame(data.get(t, []))
                if not df.empty:
                    df.to_sql(t, c, if_exists="append", index=False)
        st.success("Ripristinato.")
        st.rerun()
