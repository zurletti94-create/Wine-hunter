import json
import re
import sqlite3
import threading
import time
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

# Siti di partenza: (nome, paese, URL di ricerca con {q}, attivo).
# Verificati: Tannico, Callmewine. Gli altri sono da controllare con "Prova" o con i link manuali.
SHOPS_VERSION = 3
OLD_DEFAULTS = ["Millésima", "Vinatis", "Vinissimus", "Bodeboca", "Weinfreunde", "Hawesko", "Wein & Co"]
DEFAULT_SHOPS = [
    ("Tannico", "Italia", "https://www.tannico.it/search?q={q}", 1),
    ("Callmewine", "Italia", "https://www.callmewine.com/search?query={q}", 1),
    ("Gino Frutta & Bistrot", "Italia", "https://ginofruttabistrot.it/?s={q}&post_type=product", 1),
    ("Vinossimo", "Francia", "https://vinossimo.com/recherche?controller=search&s={q}", 1),
    ("Buveur de vin", "Francia", "https://buveurdevin.com/?s={q}&post_type=product", 1),
    ("La Pépinière", "Francia", "https://restaurant-cave-lapepiniere.com/products.json", 1),
    ("Les Zinzins du Vin", "Francia", "https://www.leszinzinsduvin.com/vins.php?q={q}", 0),  # URL di ricerca da trovare
    ("Kierdorf Wein", "Germania", "https://www.kierdorfwein.de/search?search={q}", 1),
    ("Weinfurore", "Germania", "https://www.weinfurore.de/suche?search={q}", 1),
    ("Winerl", "Austria", "https://www.winerl.com/en/search?controller=search&s={q}", 1),
    ("Versus", "Italia", "https://www.versus.wine/search?q={q}", 1),  # paese da confermare
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
        if c.execute("PRAGMA user_version").fetchone()[0] < SHOPS_VERSION:
            c.executemany("DELETE FROM shops WHERE name=?", [(n,) for n in OLD_DEFAULTS])
            c.executemany("INSERT INTO shops(name,country,url,enabled) VALUES(?,?,?,?) "
                          "ON CONFLICT(name) DO UPDATE SET country=excluded.country, url=excluded.url, enabled=excluded.enabled",
                          DEFAULT_SHOPS)
            c.execute(f"PRAGMA user_version={SHOPS_VERSION}")


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


def build_url(tpl, text):
    """Toglie il #frammento, aggiunge {q} se manca e inserisce il testo cercato."""
    tpl = tpl.split("#")[0].strip()
    if "products.json" in tpl:  # negozio Shopify letto via catalogo: il link manuale usa la ricerca del sito
        tpl = "/".join(tpl.split("/")[:3]) + "/search?q={q}"
    if "{q}" not in tpl:
        tpl += "{q}"
    return tpl.replace("{q}", quote_plus(text))


def fetch(url):
    try:
        with requests.Session() as sess:
            sess.headers.update(UA)
            sess.headers["Referer"] = "/".join(url.split("/")[:3]) + "/"
            r = sess.get(url, timeout=15)
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


_CAT, _LOCK = {}, threading.Lock()


def shopify_catalog(url):
    """Negozi Shopify: scarica una volta (cache 30 min) l'elenco prodotti da /products.json."""
    with _LOCK:
        hit = _CAT.get(url)
        if hit and time.time() - hit[0] < 1800:
            return hit[1], hit[2]
        origin, items, err = "/".join(url.split("/")[:3]), [], None
        for page in range(1, 9):
            txt, err = fetch(f"{origin}/products.json?limit=250&page={page}")
            if err:
                break
            try:
                prods = json.loads(txt).get("products", [])
            except Exception:
                err = "risposta non valida (catalogo non pubblico?)"
                break
            for p in prods:
                vs = p.get("variants", [])
                ok = [v for v in vs if v.get("available")]
                prices = [to_float(v.get("price")) for v in (ok or vs)]
                prices = [x for x in prices if x]
                if prices:
                    title = f'{p.get("title", "")} - {p.get("vendor", "")}' + ("" if ok else " (non disponibile)")
                    items.append((title, min(prices), "EUR", f'{origin}/products/{p.get("handle", "")}'))
            if len(prods) < 250:
                break
        err = err if not items else None
        _CAT[url] = (time.time(), items, err)
        return items, err


def matches(text, toks, strict=True):
    """strict: servono tutte le parole (o quasi, se sono molte); altrimenti ne basta una."""
    if not toks:
        return True
    hits = sum(x in text for x in toks)
    if not strict:
        return hits >= 1
    return hits >= (len(toks) if len(toks) <= 3 else len(toks) - 1)


def search_one(w, s):
    if "products.json" in s["url"]:
        raw, err = shopify_catalog(s["url"])
    else:
        html, err = fetch(build_url(s["url"], f'{w["producer"]} {w["wine"]}'))
        raw = [] if err else parse(html, s["url"])
    if err:
        return w["wine_id"], s["id"], [], f'{s["name"]}: {err}', 0
    ptok, wtok, res, seen = tokens(w["producer"]), tokens(w["wine"]), [], set()
    for title, price, cur, link in raw:
        t = norm(f"{title} {link}")  # titolo + indirizzo del prodotto (di solito contiene produttore e vino)
        if price is None or not 3 <= price <= 20000 or link in seen:
            continue
        if not matches(t, ptok) or not matches(t, wtok, strict=False):
            continue
        seen.add(link)
        res.append((title[:200], price, cur, link))
    return w["wine_id"], s["id"], res, None, len(raw)


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
                bar, errs, stats = st.progress(0.0), [], {}
                with ThreadPoolExecutor(8) as ex:
                    for i, (wid, sid, res, err, raw) in enumerate(ex.map(lambda j: search_one(*j), jobs), 1):
                        save_offers(wid, sid, res)
                        a, b = stats.get(sid, (0, 0))
                        stats[sid] = (a + raw, b + len(res))
                        if err:
                            errs.append(err)
                        bar.progress(i / len(jobs))
                st.success("Ricerca completata.")
                names = dict(zip(shops["id"], shops["name"]))
                st.caption("Riepilogo per sito: se i prodotti sulla pagina sono molti ma i pertinenti zero, il sito ignora la ricerca (URL sbagliato o risultati caricati via JavaScript).")
                st.dataframe(pd.DataFrame([{"Sito": names.get(k), "Prodotti sulla pagina": a, "Pertinenti": b}
                                           for k, (a, b) in stats.items()]), hide_index=True)
                if errs:
                    with st.expander(f"{len(errs)} errori"):
                        st.write("\n\n".join(sorted(set(errs))[:30]))
        with st.expander("🔗 Link di ricerca manuali (per i siti che bloccano l'app)"):
            ms = st.multiselect("Sito", sorted(shops["name"]), key="ms")
            mp = st.multiselect("Produttore", sorted(wines["producer"].unique()), key="mp")
            rows = [
                {"Sito": s["name"], "Paese": s["country"], "Produttore": w["producer"], "Vino": w["wine"],
                 "Cerca": build_url(s["url"], f'{w["producer"]} {w["wine"]}')}
                for s in shops.to_dict("records") if not ms or s["name"] in ms
                for w in wines.to_dict("records") if not mp or w["producer"] in mp
            ]
            st.caption(f"{len(rows)} link")
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values(["Sito", "Produttore", "Vino"]),
                             use_container_width=True, hide_index=True,
                             column_config={"Cerca": st.column_config.LinkColumn("Link", display_text="Apri")})
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
            turl = sh[sh["name"] == tsite].iloc[0]["url"]
            if "products.json" in turl:
                items, err = shopify_catalog(turl)
            else:
                url = build_url(turl, tq)
                html, err = fetch(url)
                items = [] if err else parse(html, url)
            if err:
                st.error(err)
            else:
                qt = tokens(tq)
                rel = [i for i in items if all(x in norm(f"{i[0]} {i[3]}") for x in qt)]
                st.write(f"{len(items)} prodotti sulla pagina, {len(rel)} pertinenti a «{tq}»")
                if items and not rel:
                    st.warning("Il sito sta ignorando la ricerca: mostra prodotti generici. Controlla l'URL (il parametro di ricerca) oppure usa i link manuali.")
                elif not items:
                    st.info("Nessun prodotto: il sito potrebbe caricare i risultati via JavaScript o bloccare i bot.")
                if rel:
                    st.dataframe(pd.DataFrame(rel, columns=["Titolo", "Prezzo", "Valuta", "Link"]), hide_index=True)

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
