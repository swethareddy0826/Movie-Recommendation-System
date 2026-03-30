import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

# ── Known actresses present in this dataset ────────────────────────────────────
# Built from manual inspection of all 343 unique cast members in mdr_dataset
KNOWN_ACTRESSES = {
    "Savitri","Anjali Devi","Bhanumathi","Malathi","Vanisri","Nargis","Meena",
    "Samantha","Anushka","Anushka Shetty","Kajal Aggarwal","Kajol","Keerthy Suresh",
    "Tamannaah","Nithya Menen","Vijayashanti","Swathi Reddy","Kamalinee","Alia",
    "Deepika","Madhuri","Kareena","Katrina","Sridevi","Vidya Balan","Tabu",
    "Kangana Ranaut","Kriti Sanon","Sonakshi","Sonakshi Sinha","Janhvi Kapoor",
    "Sara Ali Khan","Sharvari","Sharvari Wagh","Mrunal","Nazriya","Nargis",
    "Scarlett Johansson","Sigourney Weaver","Michelle Yeoh","Audrey Tautou",
    "Catherine Deneuve","Sophia Loren","Gong Li","Kim Min-hee","Kim Tae-ri",
    "Noémie Merlant","Adèle Exarchopoulos","Sandra Hüller","Anya Taylor-Joy",
    "Amy Poehler","Ariana Grande","Auli'i Cravalho","Ivana Baquero",
    "Jessica Harper","Junko Iwao","Kimiko Ikegami","Lubna Azabal","Nobuko Miyamoto",
    "Oulaya Amamra","Salma Hayek","Sara Arjun","Sheila Vand","Sylvia Bataille",
    "Waad Mohammed","Bérénice Bejo","Belén Rueda","F. Montenegro",
    "Adèle Exarchopoulos","Nithya Menen","Swathi Reddy","Newcomers",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
:root {
    --bg:#f0ebff;--card:#ffffff;--purple:#7c3aed;--purple2:#9f67fa;
    --teal:#0d9488;--coral:#f43f5e;--amber:#f59e0b;--green:#10b981;
    --text:#1e1b4b;--text2:#4c4899;--muted:#7c7ab8;--border:#ddd6fe;--radius:14px;
    --shadow:0 4px 20px rgba(124,58,237,0.09);
}
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif !important;color:var(--text) !important;}
.stApp{background:linear-gradient(135deg,#3b0764 0%,#6d28d9 35%,#9333ea 60%,#db2777 85%,#be185d 100%) !important;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stHeader"],
[data-testid="stDecoration"],[data-testid="stStatusWidget"],
[data-testid="stMainMenu"],.stDeployButton{display:none !important;height:0 !important;}
*{box-sizing:border-box;}
.stApp > div,.stApp > div > div,
[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],[data-testid="stMain"] > div,
[data-testid="stMainBlockContainer"],
.block-container,div[class*="block-container"]{padding-top:0 !important;margin-top:0 !important;}
.block-container{padding:0 !important;margin:0 auto !important;max-width:100% !important;}

.title-banner{background:transparent !important;padding:2.8rem 3rem 2.2rem !important;margin:0 !important;width:100% !important;text-align:center;}
.title-main{
    font-family:'Playfair Display',serif;font-size:4.6rem;font-weight:900;
    background:linear-gradient(135deg,#ff6b35 0%,#f7c59f 25%,#ffeb3b 50%,#ff9800 75%,#ff6b35 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    background-size:300% auto;margin:0;line-height:1.05;letter-spacing:-.02em;
    animation:titleshine 4s ease infinite;
    filter:drop-shadow(0 6px 24px rgba(255,107,53,0.6)) drop-shadow(0 2px 8px rgba(0,0,0,0.5));
}
@keyframes titleshine{0%{background-position:0% center;}50%{background-position:150% center;}100%{background-position:0% center;}}
.title-sub{
    font-family:'Plus Jakarta Sans',sans-serif;font-size:1.05rem;color:#fbbf24;
    -webkit-text-fill-color:#fbbf24;margin-top:.9rem;font-weight:800;letter-spacing:.38em;
    text-transform:uppercase;position:relative;display:inline-block;
    text-shadow:0 0 25px rgba(251,191,36,0.9),0 2px 10px rgba(0,0,0,0.5);
}
.title-sub::before,.title-sub::after{content:'';position:absolute;top:50%;width:50px;height:2px;background:#fbbf24;}
.title-sub::before{right:calc(100% + 10px);}
.title-sub::after{left:calc(100% + 10px);}

[data-testid="stMain"]{
    background:
        radial-gradient(ellipse at 0% 0%,#c084fc44 0%,transparent 50%),
        radial-gradient(ellipse at 100% 0%,#f472b644 0%,transparent 50%),
        radial-gradient(ellipse at 100% 100%,#818cf844 0%,transparent 50%),
        radial-gradient(ellipse at 0% 100%,#34d39944 0%,transparent 50%),
        linear-gradient(145deg,#f0e6ff 0%,#e8d5f5 20%,#ddd6fe 45%,#fce7f3 70%,#ffe4f0 100%) !important;
}
[data-testid="stMainBlockContainer"]{background:transparent !important;}
.block-container{
    background:transparent !important;padding:2rem 2.5rem 3rem !important;
    margin:0 auto !important;max-width:1400px !important;position:relative;z-index:1;
}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#3b0764 0%,#6d28d9 40%,#7c3aed 100%) !important;
    border-right:none !important;box-shadow:3px 0 15px rgba(0,0,0,0.15) !important;
}
[data-testid="stSidebar"] *{color:#fff !important;-webkit-text-fill-color:#fff !important;}
[data-testid="stSidebar"] .stSelectbox label{color:rgba(255,255,255,0.6) !important;font-size:.7rem !important;text-transform:uppercase;letter-spacing:.1em;font-weight:700;}
[data-testid="stSidebar"] .stSelectbox>div>div{background:rgba(255,255,255,0.15) !important;border:1px solid rgba(255,255,255,0.25) !important;border-radius:10px !important;}
.sidebar-stats{margin:1.4rem .3rem 0;background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.2);border-radius:14px;padding:1rem 1.1rem;}
.sidebar-stat-row{display:flex;justify-content:space-between;align-items:center;padding:.38rem 0;}
.sidebar-stat-row+.sidebar-stat-row{border-top:1px solid rgba(255,255,255,0.1);}
.sidebar-stat-lbl{font-size:.72rem;color:rgba(255,255,255,0.55);font-weight:500;}
.sidebar-stat-val{font-size:.82rem;font-weight:700;color:#fff;}
.sec-head{font-family:'Playfair Display',serif;font-size:1.45rem;font-weight:700;color:var(--text);border-left:4px solid var(--purple);padding-left:.8rem;margin-bottom:1rem;}
h1,h2,h3,h4{color:#1e1b4b !important;}
.stat-card{flex:1;min-width:120px;background:rgba(255,255,255,0.85);border:1.5px solid rgba(168,85,247,0.2);border-radius:var(--radius);padding:1.1rem 1.3rem;text-align:center;box-shadow:0 6px 20px rgba(124,58,237,0.1);backdrop-filter:blur(6px);}
.stat-num{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;background:linear-gradient(135deg,var(--purple),var(--coral));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1;}
.stat-label{color:var(--muted);font-size:.73rem;text-transform:uppercase;letter-spacing:.07em;margin-top:.3rem;font-weight:600;}
.info-box{background:rgba(239,246,255,0.9);border:1.5px solid #93c5fd;border-radius:10px;padding:.8rem 1.1rem;color:#1e40af;font-size:.85rem;margin:.7rem 0;}
.warn-box{background:rgba(255,251,235,0.9);border:1.5px solid #fcd34d;border-radius:10px;padding:.8rem 1.1rem;color:#92400e;font-size:.85rem;margin:.7rem 0;}
.ok-box{background:rgba(240,253,244,0.9);border:1.5px solid #bbf7d0;border-radius:10px;padding:.8rem 1.1rem;color:#14532d;font-size:.85rem;margin:.7rem 0;}
.rec-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:.9rem;}
.rec-card{background:rgba(255,255,255,0.85);border:1.5px solid var(--border);border-radius:var(--radius);padding:1.1rem;box-shadow:var(--shadow);transition:transform .2s,box-shadow .2s;backdrop-filter:blur(6px);}
.rec-card:hover{transform:translateY(-4px);box-shadow:0 10px 28px rgba(124,58,237,0.14);}
.rec-rank{font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--purple2);margin-bottom:.35rem;}
.rec-title{font-size:.93rem;font-weight:700;color:var(--text);margin-bottom:.4rem;line-height:1.3;}
.rec-meta{font-size:.76rem;color:var(--muted);margin-top:2px;}
.sel-card{background:linear-gradient(135deg,rgba(245,240,255,0.95),rgba(253,242,248,0.95));border:2px solid var(--purple);border-radius:var(--radius);padding:1.4rem 1.8rem;margin-bottom:1.4rem;box-shadow:0 6px 24px rgba(124,58,237,0.11);}
.sel-title{font-family:'Playfair Display',serif;font-size:1.55rem;font-weight:700;color:var(--purple);margin-bottom:.5rem;}
.pill{display:inline-block;background:#ede6ff;border:1px solid var(--border);border-radius:20px;padding:3px 13px;font-size:.77rem;color:var(--text2);margin:3px 3px 3px 0;font-weight:500;}
.prep-card{background:rgba(255,255,255,0.85);border:1.5px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.5rem;margin-bottom:.9rem;box-shadow:var(--shadow);border-left:5px solid var(--purple);backdrop-filter:blur(6px);}
.prep-card.teal{border-left-color:var(--teal);}
.prep-card.coral{border-left-color:var(--coral);}
.prep-card.green{border-left-color:var(--green);}
.prep-card.amber{border-left-color:var(--amber);}
.prep-num{font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:700;color:var(--purple);line-height:1;}
.prep-num.teal{color:var(--teal);}
.prep-num.coral{color:var(--coral);}
.prep-num.green{color:var(--green);}
.prep-lbl{font-size:.75rem;text-transform:uppercase;color:var(--muted);letter-spacing:.07em;font-weight:600;margin-top:.25rem;}
.badge-ok{background:#d1fae5;color:#065f46;border-radius:20px;padding:2px 10px;font-size:.72rem;font-weight:700;}
.badge-warn{background:#fef3c7;color:#92400e;border-radius:20px;padding:2px 10px;font-size:.72rem;font-weight:700;}
.stSelectbox>div>div,.stTextInput>div>div>input{background:rgba(255,255,255,0.88) !important;border:1.5px solid #c4b5fd !important;border-radius:10px !important;color:#1e1b4b !important;}
.stSelectbox label{color:#4c4899 !important;font-weight:600;}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#a855f7) !important;color:#fff !important;border:none !important;border-radius:10px !important;font-weight:700 !important;padding:.55rem 2rem !important;font-size:.9rem !important;box-shadow:0 4px 14px rgba(124,58,237,0.28) !important;}
.stButton>button:hover{opacity:.87 !important;}
.feat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:0 0 1.5rem;}
.feat-card{background:rgba(255,255,255,0.82);border:1.5px solid rgba(168,85,247,0.25);border-radius:16px;padding:1.5rem 1.2rem;text-align:center;box-shadow:var(--shadow);transition:transform .25s,box-shadow .25s;position:relative;overflow:hidden;backdrop-filter:blur(8px);}
.feat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,var(--c1),var(--c2));}
.feat-card:hover{transform:translateY(-5px);box-shadow:0 14px 36px rgba(124,58,237,0.14);}
.feat-icon{font-size:2.2rem;margin-bottom:.7rem;}
.feat-num{font-family:'Playfair Display',serif;font-size:1rem;font-weight:900;color:#7c3aed;margin-bottom:.3rem;}
.feat-title{font-size:.88rem;font-weight:700;color:#1e1b4b;margin-bottom:.3rem;}
.feat-desc{font-size:.76rem;color:#7c7ab8;line-height:1.5;}
.upload-zone{background:linear-gradient(135deg,rgba(245,240,255,0.9),rgba(253,242,248,0.9));border:2px dashed #c4b5fd;border-radius:18px;padding:2rem 1.5rem;margin:0 0 1rem;text-align:center;}
.home-stat-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.4rem 0;}
.hstat{background:linear-gradient(135deg,var(--c1),var(--c2));border-radius:14px;padding:1.2rem;text-align:center;box-shadow:0 6px 20px rgba(0,0,0,0.1);}
.hstat-num{font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;color:#fff;}
.hstat-lbl{font-size:.7rem;color:rgba(255,255,255,0.8);text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-top:.2rem;}
.divider{height:2px;background:linear-gradient(90deg,transparent,#c084fc,#f472b6,transparent);margin:1.2rem 0;border-radius:2px;}
.scroll-hint{text-align:center;color:var(--muted);font-size:.8rem;margin-top:.5rem;animation:bounce 1.5s infinite;}
@keyframes bounce{0%,100%{transform:translateY(0);}50%{transform:translateY(4px);}}
</style>
""", unsafe_allow_html=True)

CLUSTER_COLORS = ["#7c3aed","#f43f5e","#10b981","#0ea5e9","#320fce"]

# ── Helpers ────────────────────────────────────────────────────────────────────
TEXT_COLS = {"title","genres","keywords","language","director","main_cast",
             "mood_category","watch_time_category","age_group","family_friendly","movie_id"}

def clean_numeric(df):
    def parse_val(v):
        if pd.isna(v): return np.nan
        s = str(v).strip().replace(",","")
        try: return float(s)
        except:
            su = s.upper()
            try:
                if su.endswith("M"): return float(su[:-1])*1_000_000
                if su.endswith("K"): return float(su[:-1])*1_000
                if su.endswith("B"): return float(su[:-1])*1_000_000_000
            except: pass
            return np.nan
    for col in df.select_dtypes(include=["object","string"]).columns:
        if col.lower() in TEXT_COLS: continue
        trial = df[col].apply(parse_val)
        non_null = df[col].notna().sum()
        if non_null > 0 and trial.notna().sum()/non_null > 0.8:
            df[col] = trial
    return df

def parse_cast(cell):
    """Split comma-separated cast, strip role notes like (Triple Role)."""
    names = []
    for n in str(cell).split(","):
        n = n.strip().split("(")[0].strip()
        if n: names.append(n)
    return names

def split_genres(cell):
    parts = []
    for p in str(cell).replace("|","/").split("/"):
        p = p.strip()
        if p: parts.append(p)
    return parts

def build_feat(df):
    df = clean_numeric(df.copy())
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    num = [c for c in ["imdb_rating","vote_count","popularity_score","runtime_minutes",
                        "release_year","hidden_gem_score","group_watch_score"] if c in df.columns]
    f = df[num].apply(pd.to_numeric, errors="coerce").fillna(0).copy()
    if "genres" in df.columns:
        for g in sorted(set(g for cell in df["genres"].dropna() for g in split_genres(cell))):
            f[f"g_{g}"] = df["genres"].fillna("").str.contains(g, case=False, regex=False).astype(int)
    if "language" in df.columns:
        for lg in sorted(df["language"].dropna().unique()):
            f[f"l_{lg}"] = (df["language"].fillna("") == lg).astype(int)
    if "mood_category" in df.columns:
        for m in sorted(df["mood_category"].dropna().unique()):
            f[f"m_{m}"] = (df["mood_category"].fillna("") == m).astype(int)
    if "watch_time_category" in df.columns:
        for w in sorted(df["watch_time_category"].dropna().unique()):
            f[f"w_{w}"] = (df["watch_time_category"].fillna("") == w).astype(int)
    f = f.apply(pd.to_numeric, errors="coerce").fillna(0)
    return StandardScaler().fit_transform(f)

def make_cluster_names(df):
    names = {}
    for cid in range(5):
        subset = df[df["cluster"] == cid]
        if "genres" in subset.columns and not subset.empty:
            gc = {}
            for cell in subset["genres"].dropna():
                for g in split_genres(str(cell)):
                    gc[g] = gc.get(g,0)+1
            top = sorted(gc, key=gc.get, reverse=True)
            names[cid] = " & ".join(top[:2]) if len(top)>=2 else (top[0] if top else f"Cluster {cid}")
        else:
            names[cid] = f"Cluster {cid}"
    return names

def clean_dataset(df):
    if "imdb_rating" in df.columns: df = df.sort_values("imdb_rating", ascending=False)
    return df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

@st.cache_data(show_spinner="Training ML model...")
def load_ml(file_bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = clean_numeric(df)
    for c in df.select_dtypes(include=np.number).columns: df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include="object").columns:  df[c] = df[c].fillna("Unknown")
    df = clean_dataset(df)
    sc = build_feat(df)
    km = KMeans(n_clusters=5, random_state=42, n_init=3, max_iter=100, algorithm="lloyd")
    df["cluster"] = km.fit_predict(sc)
    sim = cosine_similarity(sc).astype(np.float32)
    return df, sc, sim, make_cluster_names(df)

@st.cache_data(show_spinner=False)
def get_raw(file_bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]

def get_cast_lists(df):
    """Return actors (male) and actresses (female) from dataset cast column."""
    all_cast = set()
    for cell in df["main_cast"].dropna():
        for name in parse_cast(str(cell)):
            all_cast.add(name)
    actresses = sorted(n for n in all_cast if n in KNOWN_ACTRESSES)
    actors    = sorted(n for n in all_cast if n not in KNOWN_ACTRESSES)
    return actors, actresses

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding:1.2rem .5rem .5rem;font-size:.7rem;color:rgba(255,255,255,0.55);text-transform:uppercase;letter-spacing:.12em;font-weight:700">Navigation</div>', unsafe_allow_html=True)
    page = st.selectbox("", ["Home & Upload","Preprocessing Report","Movie Recommendations","Cluster Visualization","Top Rated Movies"])
    st.markdown("""
    <div class="sidebar-stats">
        <div class="sidebar-stat-row"><span class="sidebar-stat-lbl">Algorithm</span><span class="sidebar-stat-val">K-Means</span></div>
        <div class="sidebar-stat-row"><span class="sidebar-stat-lbl">Similarity</span><span class="sidebar-stat-val">Cosine</span></div>
        <div class="sidebar-stat-row"><span class="sidebar-stat-lbl">Clusters</span><span class="sidebar-stat-val">5 Groups</span></div>
        <div class="sidebar-stat-row"><span class="sidebar-stat-lbl">Results</span><span class="sidebar-stat-val">Top 5</span></div>
    </div>""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-banner">
    <div class="title-main">CineMatch</div>
    <div class="title-sub">Movie Recommendation System</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HOME & UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home & Upload":
    if "fb" not in st.session_state:
        st.markdown("""
        <div class="feat-grid">
            <div class="feat-card" style="--c1:#7c3aed;--c2:#a855f7">
                <div class="feat-icon">📂</div><div class="feat-num">Step 1</div>
                <div class="feat-title">Upload Dataset</div>
                <div class="feat-desc">Upload your movie CSV — ratings, genres, language, cast &amp; more</div>
            </div>
            <div class="feat-card" style="--c1:#0d9488;--c2:#14b8a6">
                <div class="feat-icon">🔬</div><div class="feat-num">Step 2</div>
                <div class="feat-title">Preprocessing</div>
                <div class="feat-desc">Auto cleaning — missing values, duplicates and quality report</div>
            </div>
            <div class="feat-card" style="--c1:#f43f5e;--c2:#fb7185">
                <div class="feat-icon">🔵</div><div class="feat-num">Step 3</div>
                <div class="feat-title">K-Means Clustering</div>
                <div class="feat-desc">Movies grouped into 5 clusters based on genres, ratings &amp; features</div>
            </div>
            <div class="feat-card" style="--c1:#f59e0b;--c2:#fbbf24">
                <div class="feat-icon">🎯</div><div class="feat-num">Step 4</div>
                <div class="feat-title">Recommendations</div>
                <div class="feat-desc">Cosine Similarity surfaces the 5 most similar movies for you</div>
            </div>
        </div>
        <div class="upload-zone">
            <div style="font-size:2.8rem;margin-bottom:.6rem">🎬</div>
            <div style="font-size:1.1rem;font-weight:700;color:#1e1b4b;margin-bottom:.3rem">Upload Your Movie Dataset</div>
            <div style="font-size:.8rem;color:#7c7ab8">Drop a CSV file below to get started</div>
        </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    if uploaded:
        st.session_state["fb"] = uploaded.read()
        raw_df = get_raw(st.session_state["fb"])
        yr_min = int(raw_df["release_year"].min()) if "release_year" in raw_df.columns else "—"
        yr_max = int(raw_df["release_year"].max()) if "release_year" in raw_df.columns else "—"
        n_lang = raw_df["language"].nunique() if "language" in raw_df.columns else "—"
        total  = raw_df.shape[0]
        st.markdown(f"""
        <div class="ok-box" style="font-size:.9rem;margin-bottom:1.2rem">
            ✅ Dataset loaded — <b>{total:,} movies</b> across <b>{raw_df.shape[1]}</b> columns. Use the sidebar to navigate.
        </div>
        <div class="home-stat-strip">
            <div class="hstat" style="--c1:#7c3aed;--c2:#a855f7"><div class="hstat-num">{total:,}</div><div class="hstat-lbl">Total Movies</div></div>
            <div class="hstat" style="--c1:#0d9488;--c2:#14b8a6"><div class="hstat-num">{n_lang}</div><div class="hstat-lbl">Languages</div></div>
            <div class="hstat" style="--c1:#f43f5e;--c2:#fb7185"><div class="hstat-num">{raw_df.shape[1]}</div><div class="hstat-lbl">Columns</div></div>
            <div class="hstat" style="--c1:#f59e0b;--c2:#fbbf24"><div class="hstat-num">{yr_min}–{yr_max}</div><div class="hstat-lbl">Year Range</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  GUARD
# ══════════════════════════════════════════════════════════════════════════════
elif "fb" not in st.session_state:
    st.markdown('<div class="warn-box" style="font-size:.95rem;padding:1.2rem 1.5rem">⚠️ Go to <b>Home &amp; Upload</b> and upload your CSV file first.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGES
# ══════════════════════════════════════════════════════════════════════════════
else:
    df, scaled, sim, CLUSTER_NAMES = load_ml(st.session_state["fb"])
    rdf = get_raw(st.session_state["fb"])

    # ══════════════════════════════════════════════════════════════════════════
    #  PREPROCESSING REPORT
    # ══════════════════════════════════════════════════════════════════════════
    if page == "Preprocessing Report":
        st.markdown('<div class="sec-head">Preprocessing Report</div>', unsafe_allow_html=True)

        st.markdown("#### Step 1 — Dataset Shape")
        c1,c2,c3,c4 = st.columns(4)
        num_num = rdf.select_dtypes(include=np.number).shape[1]
        for col,num,lbl,cls in zip([c1,c2,c3,c4],
            [f"{rdf.shape[0]:,}", rdf.shape[1], num_num, rdf.select_dtypes(include="object").shape[1]],
            ["Total Rows","Total Columns","Numeric Columns","Text Columns"],["","teal","coral","green"]):
            with col:
                st.markdown(f'<div class="prep-card {cls}"><div class="prep-num {cls}">{num}</div><div class="prep-lbl">{lbl}</div></div>', unsafe_allow_html=True)
        with st.expander("View Raw Dataset (first 5 rows)"):
            st.dataframe(rdf.head(), use_container_width=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown("#### Step 2 — Missing Values")
        miss   = rdf.isnull().sum()
        miss_p = (miss/len(rdf)*100).round(2)
        total_miss = int(miss.sum()); cols_miss = int((miss>0).sum())
        complete   = round((1-total_miss/rdf.size)*100, 2)
        c1,c2,c3 = st.columns(3)
        with c1:
            b=("badge-ok" if total_miss==0 else "badge-warn"); cs=("green" if total_miss==0 else "amber")
            st.markdown(f'<div class="prep-card {cs}"><div class="prep-num {cs}">{total_miss:,}</div><div class="prep-lbl">Total Missing Cells &nbsp;<span class="{b}">{"Clean" if total_miss==0 else "Found"}</span></div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="prep-card teal"><div class="prep-num teal">{cols_miss}</div><div class="prep-lbl">Columns with Missing Values</div></div>', unsafe_allow_html=True)
        with c3:
            b2=("badge-ok" if complete>=98 else "badge-warn"); cs2=("green" if complete>=98 else "amber")
            st.markdown(f'<div class="prep-card {cs2}"><div class="prep-num {cs2}">{complete}%</div><div class="prep-lbl">Data Completeness &nbsp;<span class="{b2}">{"Good" if complete>=98 else "Check"}</span></div></div>', unsafe_allow_html=True)
        miss_df = pd.DataFrame({
            "Column": miss.index, "Missing Count": miss.values, "Missing %": miss_p.values,
            "Status": ["✅ Clean" if v==0 else ("⚠️ Minor (<5%)" if p<5 else "🔴 High (≥5%)") for v,p in zip(miss.values,miss_p.values)]
        }).sort_values("Missing Count", ascending=False)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)
        st.markdown('<div class="ok-box">✅ No missing values — dataset is complete.</div>' if total_miss==0
                    else f'<div class="warn-box">⚠️ <b>{total_miss:,} missing values</b> across <b>{cols_miss}</b> column(s). Auto-filled during ML processing.</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown("#### Step 3 — Duplicate Rows")
        dup_mask  = rdf.duplicated(); dup_count = int(dup_mask.sum())
        dup_title = int(rdf.duplicated(subset=["title"]).sum()) if "title" in rdf.columns else 0
        after_drop = rdf.shape[0] - dup_count
        c1,c2,c3 = st.columns(3)
        with c1:
            b=("badge-ok" if dup_count==0 else "badge-warn"); cs=("green" if dup_count==0 else "coral")
            st.markdown(f'<div class="prep-card {cs}"><div class="prep-num {cs}">{dup_count}</div><div class="prep-lbl">Exact Duplicate Rows &nbsp;<span class="{b}">{"None" if dup_count==0 else "Found"}</span></div></div>', unsafe_allow_html=True)
        with c2:
            b2=("badge-ok" if dup_title==0 else "badge-warn"); cs2=("green" if dup_title==0 else "amber")
            st.markdown(f'<div class="prep-card {cs2}"><div class="prep-num {cs2}">{dup_title}</div><div class="prep-lbl">Duplicate Titles &nbsp;<span class="{b2}">{"None" if dup_title==0 else "Found"}</span></div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="prep-card teal"><div class="prep-num teal">{after_drop:,}</div><div class="prep-lbl">Rows After Dedup</div></div>', unsafe_allow_html=True)
        if dup_count > 0:
            with st.expander(f"👁 View {dup_count} exact duplicate rows"):
                st.dataframe(rdf[dup_mask], use_container_width=True)
        st.markdown('<div class="ok-box">✅ No exact duplicate rows.</div>' if dup_count==0
                    else f'<div class="warn-box">⚠️ {dup_count} exact duplicates removed automatically.</div>', unsafe_allow_html=True)
        if dup_title > 0:
            with st.expander(f"👁 View {dup_title} duplicate title entries"):
                st.dataframe(rdf[rdf.duplicated(subset=["title"],keep=False)].sort_values("title"), use_container_width=True, hide_index=True)
        st.markdown('<div class="ok-box">✅ No duplicate movie titles.</div>' if dup_title==0
                    else f'<div class="warn-box">⚠️ {dup_title} rows share duplicate titles. Highest-rated version kept.</div>', unsafe_allow_html=True)

        clean_rows = len(df)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#7c3aed,#0d9488);border-radius:14px;padding:1.3rem 2rem;margin-top:1.2rem">
            <div style="color:#fff;font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700">Preprocessing Summary</div>
            <div style="color:rgba(255,255,255,.88);font-size:.85rem;margin-top:.5rem;line-height:1.9">
                📂 Uploaded: <b>{rdf.shape[0]:,} rows</b> &nbsp;·&nbsp; 🔴 Exact dups: <b>{dup_count}</b> &nbsp;·&nbsp;
                📋 Title dups: <b>{dup_title}</b> &nbsp;·&nbsp; ❓ Missing: <b>{total_miss:,}</b><br>
                ✅ After cleaning: <b>{clean_rows:,} unique movies</b> ready for ML
            </div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  MOVIE RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Movie Recommendations":
        st.markdown('<div class="sec-head">Movie Recommendations</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">All filter options are loaded directly from your dataset. Select preferences and click <b>Get Recommendations</b>.</div>', unsafe_allow_html=True)

        # Build ALL options strictly from dataset values
        genre_list = sorted(set(g for cell in df["genres"].dropna() for g in split_genres(str(cell)))) if "genres" in df.columns else []
        lang_list  = sorted(df["language"].dropna().unique().tolist()) if "language" in df.columns else []
        mood_list  = sorted(df["mood_category"].dropna().unique().tolist()) if "mood_category" in df.columns else []
        wt_list    = sorted(df["watch_time_category"].dropna().unique().tolist()) if "watch_time_category" in df.columns else []
        dir_list   = sorted(df["director"].dropna().unique().tolist()) if "director" in df.columns else []

        # Gender-split cast from dataset
        actors_list, actresses_list = get_cast_lists(df)

        # Row 1 — Genre | Language | Mood | Watch Time
        c1, c2, c3, c4 = st.columns(4)
        with c1: genre_f = st.selectbox("Genre", ["Any Genre"] + genre_list)
        with c2: lang_f  = st.selectbox("Language", ["Any Language"] + lang_list)
        with c3: mood_f  = st.selectbox("Mood", ["Any Mood"] + mood_list)
        with c4: time_f  = st.selectbox("Watch Time", ["Any Duration"] + wt_list)

        # Row 2 — Director | Actor (male only) | Actress (female only)
        c5, c6, c7 = st.columns(3)
        with c5: dir_f     = st.selectbox("Director", ["Any Director"] + dir_list)
        with c6: actor_f   = st.selectbox("Actor", ["Any Actor"] + actors_list)
        with c7: actress_f = st.selectbox("Actress", ["Any Actress"] + actresses_list)

        # Apply filters
        fdf = df.copy()
        if genre_f   != "Any Genre"    and "genres"              in df.columns: fdf = fdf[fdf["genres"].str.contains(genre_f, case=False, na=False)]
        if lang_f    != "Any Language" and "language"            in df.columns: fdf = fdf[fdf["language"] == lang_f]
        if mood_f    != "Any Mood"     and "mood_category"       in df.columns: fdf = fdf[fdf["mood_category"] == mood_f]
        if time_f    != "Any Duration" and "watch_time_category" in df.columns: fdf = fdf[fdf["watch_time_category"] == time_f]
        if dir_f     != "Any Director" and "director"            in df.columns: fdf = fdf[fdf["director"].str.contains(dir_f, case=False, na=False)]
        if actor_f   != "Any Actor"    and "main_cast"           in df.columns: fdf = fdf[fdf["main_cast"].str.contains(actor_f, case=False, na=False)]
        if actress_f != "Any Actress"  and "main_cast"           in df.columns: fdf = fdf[fdf["main_cast"].str.contains(actress_f, case=False, na=False)]
        fdf = fdf.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

        user_gave_input = any([
            genre_f != "Any Genre", lang_f != "Any Language", mood_f != "Any Mood",
            time_f != "Any Duration", dir_f != "Any Director",
            actor_f != "Any Actor", actress_f != "Any Actress",
        ])

        if not user_gave_input:
            st.markdown('<div class="warn-box">👆 Select at least one filter above, then click <b>Get Recommendations</b>.</div>', unsafe_allow_html=True)
        elif fdf.empty:
            st.markdown('<div class="warn-box">⚠️ No movies match your filters. Try relaxing some criteria.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ok-box">✅ <b>{len(fdf)}</b> movie(s) match your filters.</div>', unsafe_allow_html=True)
            view_cols = [c for c in ["title","imdb_rating","release_year","language","genres",
                                     "mood_category","runtime_minutes","watch_time_category",
                                     "age_group","family_friendly","director","main_cast",
                                     "hidden_gem_score","group_watch_score"] if c in fdf.columns]
            with st.expander(f"👁 View all {len(fdf)} matched movies"):
                st.dataframe(fdf[view_cols].sort_values("imdb_rating", ascending=False).reset_index(drop=True),
                             use_container_width=True, hide_index=True)

            if st.button("🎯 Get Recommendations"):
                seed_row = fdf.sort_values("imdb_rating", ascending=False).iloc[0]
                selected = str(seed_row["title"])
                midx = df[df["title"] == selected].index[0]
                row  = df.iloc[midx]
                cid  = int(row.get("cluster", 0))
                yr_v = int(row["release_year"]) if not pd.isna(row.get("release_year", np.nan)) else "—"
                rt_v = int(row.get("runtime_minutes", 0))

                st.markdown(f"""
                <div class="sel-card">
                    <div style="font-size:.72rem;font-weight:700;color:#9f67fa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem">Best match for your filters</div>
                    <div class="sel-title">{selected}</div>
                    <div>
                        <span class="pill">⭐ {row.get('imdb_rating',0)}/10</span>
                        <span class="pill">📅 {yr_v}</span>
                        <span class="pill">⏱ {rt_v} min</span>
                        <span class="pill">🌐 {row.get('language','—')}</span>
                        <span class="pill">🎨 {row.get('mood_category','—')}</span>
                        <span class="pill">🔞 {row.get('age_group','—')}</span>
                        <span class="pill">👨‍👩‍👧 {row.get('family_friendly','—')}</span>
                        <span class="pill">💎 Gem {row.get('hidden_gem_score','—')}</span>
                        <span class="pill">👥 Group {row.get('group_watch_score','—')}</span>
                    </div>
                    <div style="font-size:.8rem;color:#7c7ab8;margin-top:.6rem">🎭 {row.get('genres','—')}</div>
                    <div style="font-size:.8rem;color:#7c7ab8;margin-top:.3rem">🎬 {row.get('director','—')} &nbsp;|&nbsp; 🎭 {row.get('main_cast','—')}</div>
                </div>""", unsafe_allow_html=True)

                scores  = sorted(enumerate(sim[midx]), key=lambda x:x[1], reverse=True)
                scores  = [s for s in scores if s[0] != midx][:5]
                rec_idx = [s[0] for s in scores]
                rec_pct = [round(s[1]*100,1) for s in scores]

                st.markdown('<div class="sec-head" style="margin-top:.8rem">Top 5 Recommended Movies</div>', unsafe_allow_html=True)
                cards_html = '<div class="rec-grid">'
                for rank,(idx,pct) in enumerate(zip(rec_idx,rec_pct),1):
                    r  = df.iloc[idx]
                    rc = int(r.get("cluster",0))
                    yr = int(r["release_year"]) if not pd.isna(r.get("release_year",np.nan)) else "—"
                    t  = str(r.get("title","—"))
                    cards_html += f"""
                    <div class="rec-card" style="border-top:3.5px solid {CLUSTER_COLORS[rc]}">
                        <div class="rec-rank">#{rank} &nbsp; {pct}% match</div>
                        <div class="rec-title">{t}</div>
                        <div class="rec-meta">⭐ {r.get('imdb_rating',0)}/10 &nbsp; 📅 {yr}</div>
                        <div class="rec-meta">🌐 {r.get('language','—')} &nbsp; ⏱ {int(r.get('runtime_minutes',0))} min</div>
                        <div class="rec-meta">🎨 {str(r.get('mood_category',''))}</div>
                        <div class="rec-meta" style="font-size:.71rem;margin-top:3px">🎭 {str(r.get('genres',''))[:40]}</div>
                    </div>"""
                cards_html += "</div>"
                st.markdown(cards_html, unsafe_allow_html=True)
                st.markdown('<div class="scroll-hint">&#8595; scroll for full table &#8595;</div>', unsafe_allow_html=True)

                tbl = []
                for rank,(idx,pct) in enumerate(zip(rec_idx,rec_pct),1):
                    r = df.iloc[idx]
                    tbl.append({
                        "Rank": rank, "Title": str(r.get("title","—")), "Match %": f"{pct}%",
                        "IMDB": r.get("imdb_rating","—"),
                        "Year": int(r["release_year"]) if not pd.isna(r.get("release_year",np.nan)) else "—",
                        "Runtime": f"{int(r.get('runtime_minutes',0))} min",
                        "Watch Time": str(r.get("watch_time_category","—")),
                        "Language": str(r.get("language","—")),
                        "Genres": str(r.get("genres","—")),
                        "Mood": str(r.get("mood_category","—")),
                        "Age Cert": str(r.get("age_group","—")),
                        "Family Friendly": str(r.get("family_friendly","—")),
                        "Director": str(r.get("director","—")),
                        "Cast": str(r.get("main_cast","—")),
                        "Hidden Gem Score": r.get("hidden_gem_score","—"),
                        "Group Watch Score": r.get("group_watch_score","—"),
                    })
                st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  CLUSTER VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Cluster Visualization":
        st.markdown('<div class="sec-head">Cluster Visualization</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">K-Means groups movies into 5 clusters. Hover points for full details.</div>', unsafe_allow_html=True)

        counts = df["cluster"].value_counts().sort_index()
        cols_c = st.columns(5)
        for cid in range(5):
            with cols_c[cid]:
                st.markdown(f'<div class="stat-card" style="border-color:{CLUSTER_COLORS[cid]}55"><div class="stat-num" style="background:none;-webkit-text-fill-color:{CLUSTER_COLORS[cid]};color:{CLUSTER_COLORS[cid]}">{counts.get(cid,0)}</div><div class="stat-label">{CLUSTER_NAMES[cid]}</div></div>', unsafe_allow_html=True)

        df["cluster_name"]  = df["cluster"].map(CLUSTER_NAMES)
        df["cluster_color"] = df["cluster"].map(lambda x: CLUSTER_COLORS[x])
        st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Rating vs Popularity", "Distribution"])
        with tab1:
            fig = px.scatter(
                df, x="imdb_rating", y="popularity_score",
                color="cluster_name", color_discrete_sequence=CLUSTER_COLORS,
                hover_name="title",
                hover_data={"imdb_rating":":.1f","popularity_score":":.1f",
                            "cluster_name":False,"cluster_color":False,
                            "language":True,"genres":True,"mood_category":True,
                            "release_year":True,"runtime_minutes":True,"director":True},
                labels={"imdb_rating":"IMDB Rating","popularity_score":"Popularity Score","cluster_name":"Cluster"},
                title="Movie Clusters — Rating vs Popularity", opacity=0.75,
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
            fig.update_layout(paper_bgcolor="#faf7ff",plot_bgcolor="#f0ebff",
                font=dict(family="Plus Jakarta Sans",color="#1e1b4b"),
                legend=dict(title="Cluster",bgcolor="white",bordercolor="#ddd6fe",borderwidth=1),
                hovermode="closest",margin=dict(l=40,r=20,t=50,b=40),height=450)
            fig.update_xaxes(gridcolor="#ddd6fe",zerolinecolor="#ddd6fe")
            fig.update_yaxes(gridcolor="#ddd6fe",zerolinecolor="#ddd6fe")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            cv = [counts.get(i,0) for i in range(5)]
            names = [CLUSTER_NAMES[i] for i in range(5)]
            col_bar,col_pie = st.columns(2)
            with col_bar:
                fig_bar = go.Figure(go.Bar(x=names,y=cv,marker_color=CLUSTER_COLORS,text=cv,textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Movies: %{y}<extra></extra>"))
                fig_bar.update_layout(title="Movies per Cluster",paper_bgcolor="#faf7ff",plot_bgcolor="#f0ebff",
                    font=dict(family="Plus Jakarta Sans",color="#1e1b4b"),
                    margin=dict(l=20,r=20,t=50,b=80),height=370,
                    xaxis=dict(tickangle=-20,gridcolor="#ddd6fe"),yaxis=dict(gridcolor="#ddd6fe"))
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_pie:
                fig_pie = go.Figure(go.Pie(labels=names,values=cv,
                    marker=dict(colors=CLUSTER_COLORS,line=dict(color="white",width=2)),
                    hole=0.45,hovertemplate="<b>%{label}</b><br>Movies: %{value}<br>%{percent}<extra></extra>",textinfo="percent"))
                fig_pie.update_layout(title="Cluster Share",paper_bgcolor="#faf7ff",
                    font=dict(family="Plus Jakarta Sans",color="#1e1b4b"),
                    legend=dict(bgcolor="white",bordercolor="#ddd6fe",borderwidth=1),
                    margin=dict(l=20,r=20,t=50,b=20),height=370)
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="sec-head" style="margin-top:1rem">Browse Cluster</div>', unsafe_allow_html=True)
        chosen = st.selectbox("Select a cluster", [CLUSTER_NAMES[i] for i in range(5)])
        cid_s  = [k for k,v in CLUSTER_NAMES.items() if v==chosen][0]
        show_c = [c for c in ["title","imdb_rating","popularity_score","release_year","language",
                               "genres","mood_category","watch_time_category","age_group",
                               "family_friendly","runtime_minutes","director","main_cast",
                               "hidden_gem_score","group_watch_score"] if c in df.columns]
        st.dataframe(df[df["cluster"]==cid_s][show_c].head(25), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TOP RATED MOVIES
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "Top Rated Movies":
        st.markdown('<div class="sec-head">Top Rated Movies</div>', unsafe_allow_html=True)

        c_a,c_b,c_c = st.columns(3)
        with c_a:
            lang_opts = ["All"] + sorted(df["language"].dropna().unique().tolist()) if "language" in df.columns else ["All"]
            lang_f = st.selectbox("Filter by Language", lang_opts)
        with c_b:
            genre_opts = ["All Genres"] + sorted(set(g for cell in df["genres"].dropna() for g in split_genres(str(cell)))) if "genres" in df.columns else ["All Genres"]
            genre_top = st.selectbox("Filter by Genre", genre_opts)
        with c_c:
            top_n = st.slider("Show Top N", 5, 50, 20)

        fdf_top = df.copy()
        if lang_f    != "All":        fdf_top = fdf_top[fdf_top["language"] == lang_f]
        if genre_top != "All Genres": fdf_top = fdf_top[fdf_top["genres"].str.contains(genre_top, case=False, na=False)]
        top = fdf_top.sort_values("imdb_rating", ascending=False).drop_duplicates(subset=["title"]).head(top_n)

        if top.empty:
            st.markdown('<div class="warn-box">No movies match the selected filters.</div>', unsafe_allow_html=True)
        else:
            fig_top = go.Figure(go.Bar(
                y=top["title"].str[:45].values[::-1], x=top["imdb_rating"].values[::-1],
                orientation="h",
                marker=dict(color=top["imdb_rating"].values[::-1],
                    colorscale=[[0,"#a855f7"],[0.5,"#7c3aed"],[1,"#f43f5e"]],showscale=False),
                text=[f"{v:.1f}" for v in top["imdb_rating"].values[::-1]], textposition="outside",
                hovertemplate="<b>%{y}</b><br>IMDB: %{x:.1f}<extra></extra>",
            ))
            fig_top.update_layout(
                title="Top Rated Movies by IMDB Score",paper_bgcolor="#faf7ff",plot_bgcolor="#f0ebff",
                font=dict(family="Plus Jakarta Sans",color="#1e1b4b"),
                xaxis=dict(range=[0,11],gridcolor="#ddd6fe",title="IMDB Rating"),
                yaxis=dict(gridcolor="#ddd6fe",tickfont=dict(size=10)),
                margin=dict(l=20,r=60,t=50,b=30),height=max(400,top_n*24),hovermode="y unified",
            )
            st.plotly_chart(fig_top, use_container_width=True)

            if len(top) >= 3:
                st.markdown('<div style="margin:.8rem 0 .5rem"><b style="color:#7c3aed;font-size:1rem">🏆 Top 3 Podium</b></div>', unsafe_allow_html=True)
                pc = st.columns(3)
                medals = ["🥇 1st","🥈 2nd","🥉 3rd"]
                for i,(col_,(_,row)) in enumerate(zip(pc, top.head(3).iterrows())):
                    with col_:
                        yr = int(row["release_year"]) if not pd.isna(row.get("release_year",np.nan)) else "—"
                        st.markdown(f"""
                        <div class="rec-card" style="text-align:center;border-top:3.5px solid {CLUSTER_COLORS[i]}">
                            <div style="font-size:1.1rem;font-weight:700;color:{CLUSTER_COLORS[i]}">{medals[i]}</div>
                            <div class="rec-title" style="text-align:center">{str(row['title'])}</div>
                            <div class="rec-meta">⭐ {row.get('imdb_rating',0)}/10 &nbsp; 📅 {yr}</div>
                            <div class="rec-meta">🌐 {row.get('language','—')} &nbsp; 🎭 {str(row.get('genres','—'))[:30]}</div>
                            <div class="rec-meta">💎 Gem {row.get('hidden_gem_score','—')} &nbsp; 👥 Group {row.get('group_watch_score','—')}</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
            disp = [c for c in ["title","imdb_rating","vote_count","popularity_score",
                                 "release_year","language","genres","mood_category",
                                 "runtime_minutes","watch_time_category","age_group",
                                 "family_friendly","director","main_cast",
                                 "hidden_gem_score","group_watch_score"] if c in df.columns]
            st.dataframe(top[disp].reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)

# Run:
# pip install streamlit pandas numpy scikit-learn plotly
# python -m streamlit run app.py