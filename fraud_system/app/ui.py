"""
app/ui.py

FraudZilla design system — presentation only, no business logic.

Centralises all styling so pages stay consistent and their logic stays
untouched. Inject once per run from the navigation entry (main.py) via
apply_theme(); because the entry script reruns before every page, the theme
applies to all pages automatically.

Tokens: white base, 3 brand colors + 1 CTA accent, functional status colors
kept separate (they carry meaning). Type: Sora (display) + Inter (body),
Google-Fonts loaded. Spacing on an 8pt grid. Single subtle shadow, no glow.
Motion 160ms ease, disabled under prefers-reduced-motion. Icons are inline
Lucide SVG in HTML we control (sidebar/cards) and Material Symbols in the
native nav — no emoji icons.

Honest Streamlit limit: native-widget selectors (buttons, metrics, nav links)
can shift across Streamlit versions. If a future update changes them the app
still WORKS; only the look degrades. Custom HTML blocks are fully controlled.
"""

import streamlit as st


TOKENS = {
    "ink": "#1F2A37", "muted": "#5B6876", "faint": "#8A95A1",
    "brand": "#2F8F7F", "brand_deep": "#226B5F",
    "mist": "#EEF3F5", "brand_tint": "#E4F1EE",
    "canvas": "#FAFBFC", "surface": "#FFFFFF", "line": "#E3E9ED",
    "low_fg": "#216E4E", "low_bg": "#E4F4EC",
    "med_fg": "#9A6700", "med_bg": "#FBF1DA",
}

ICONS = {
    "shield": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "shield_sm": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "receipt": '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 2v20l2-1 2 1 2-1 2 1 2-1 2 1 2-1 2 1V2l-2 1-2-1-2 1-2-1-2 1-2-1-2 1z"/><path d="M8 7h8M8 11h8M8 15h5"/></svg>',
    "user": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></svg>',
    "chart": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><rect x="7" y="11" width="3" height="6"/><rect x="12" y="7" width="3" height="10"/><rect x="17" y="13" width="3" height="4"/></svg>',
    "search": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>',
    "check": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    "arrow": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
}


def icon(name: str) -> str:
    return ICONS.get(name, "")


def apply_theme() -> None:
    t = TOKENS
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        :root {{
          --ink:{t['ink']}; --muted:{t['muted']}; --faint:{t['faint']};
          --brand:{t['brand']}; --brand-deep:{t['brand_deep']};
          --mist:{t['mist']}; --brand-tint:{t['brand_tint']};
          --canvas:{t['canvas']}; --surface:{t['surface']}; --line:{t['line']};
        }}

        html, body, [class*="css"], .stApp {{
          font-family:'Inter',system-ui,sans-serif; color:var(--ink);
        }}
        .stApp {{ background:var(--canvas); }}
        .block-container {{ padding-top:40px; max-width:1160px; }}
        h1,h2,h3,.fz-display {{ font-family:'Sora',sans-serif; letter-spacing:-0.01em; }}

        /* ---------- Sidebar branding ---------- */
        [data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--line); }}
        .fz-brand {{ display:flex; align-items:center; gap:11px; padding:8px 4px 16px; }}
        .fz-brand .mark {{ width:38px; height:38px; border-radius:10px; background:var(--brand-tint);
          color:var(--brand); display:flex; align-items:center; justify-content:center; }}
        .fz-brand .name {{ font-family:'Sora'; font-weight:700; font-size:18px; line-height:1; }}
        .fz-brand .sub {{ font-size:11.5px; color:var(--faint); margin-top:3px; }}
        .fz-side-divider {{ height:1px; background:var(--line); margin:8px 0 4px; }}
        .fz-side-status {{ display:flex; align-items:center; gap:9px; padding:10px 8px; margin-top:8px;
          font-size:12.5px; color:var(--muted); border-top:1px solid var(--line); }}
        .fz-side-status .led {{ width:9px; height:9px; border-radius:999px; }}
        .fz-side-status .led.on {{ background:{t['low_fg']}; box-shadow:0 0 0 3px {t['low_bg']}; }}
        .fz-side-status .led.off {{ background:{t['med_fg']}; box-shadow:0 0 0 3px {t['med_bg']}; }}

        /* nav link active accent (fragile across versions; degrades gracefully) */
        [data-testid="stSidebarNav"] a[aria-current="page"] {{ color:var(--brand)!important; font-weight:600; }}

        /* ---------- Top status bar (overview) ---------- */
        .fz-statusbar {{ display:flex; align-items:center; justify-content:space-between;
          background:var(--surface); border:1px solid var(--line); border-radius:12px;
          padding:12px 20px; box-shadow:0 2px 8px rgba(31,42,55,0.05); margin-bottom:24px; }}
        .fz-statusbar .meta {{ display:flex; align-items:center; gap:16px; font-size:13px; color:var(--muted); }}
        .fz-statusbar .meta b {{ color:var(--ink); font-weight:600; }}
        .fz-statusbar .sep {{ width:1px; height:16px; background:var(--line); }}
        .fz-pill {{ display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:999px;
          font-size:12.5px; font-weight:600; }}
        .fz-pill.ok {{ background:{t['low_bg']}; color:{t['low_fg']}; }}
        .fz-pill.wait {{ background:{t['med_bg']}; color:{t['med_fg']}; }}

        /* ---------- Hero ---------- */
        .fz-hero {{ position:relative; overflow:hidden;
          background:linear-gradient(180deg,#FFFFFF 0%,#F6FAF9 100%);
          border:1px solid var(--line); border-radius:16px; padding:36px 40px;
          box-shadow:0 2px 8px rgba(31,42,55,0.05); margin-bottom:32px; }}
        .fz-hero::after {{ content:""; position:absolute; right:-60px; top:-60px; width:240px; height:240px;
          background:radial-gradient(circle, rgba(47,143,127,0.10), transparent 70%); }}
        .fz-hero-row {{ display:flex; align-items:center; gap:16px; position:relative; z-index:1; }}
        .fz-logo {{ width:52px; height:52px; border-radius:13px; background:var(--brand-tint);
          color:var(--brand); display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
        .fz-eyebrow {{ font-size:12px; font-weight:600; letter-spacing:0.10em; text-transform:uppercase;
          color:var(--brand); margin-bottom:4px; }}
        .fz-hero h1 {{ font-size:42px; line-height:1.05; font-weight:800; margin:0; }}
        .fz-hero p {{ font-size:16px; color:var(--muted); max-width:620px; margin-top:16px; position:relative; z-index:1; }}
        .fz-hero p strong {{ color:var(--ink); font-weight:600; }}

        /* ---------- Section ---------- */
        .fz-section {{ display:flex; align-items:baseline; justify-content:space-between; margin:0 4px 16px; }}
        .fz-section h2 {{ font-size:22px; font-weight:700; margin:0; }}
        .fz-section .hint {{ font-size:13px; color:var(--faint); }}

        /* ---------- Cards ---------- */
        .fz-grid {{ display:grid; grid-template-columns:1.4fr 1fr 1fr; gap:16px; margin-bottom:16px; }}
        .fz-grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px; }}
        .fz-card {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:24px;
          box-shadow:0 2px 8px rgba(31,42,55,0.04); display:flex; flex-direction:column;
          transition:transform 160ms ease, box-shadow 160ms ease; }}
        .fz-card:hover {{ transform:translateY(-2px); box-shadow:0 6px 18px rgba(31,42,55,0.10); }}
        .fz-card .ico {{ width:48px; height:48px; border-radius:12px; background:var(--mist); color:var(--brand);
          display:flex; align-items:center; justify-content:center; margin-bottom:16px; }}
        .fz-card h3 {{ font-size:18px; font-weight:600; margin:0 0 8px 0; }}
        .fz-card .body {{ font-size:14px; color:var(--muted); flex:1; line-height:1.5; }}
        .fz-card .go {{ margin-top:16px; font-size:13.5px; font-weight:600; color:var(--brand);
          display:flex; align-items:center; gap:6px; }}
        .fz-card.primary {{ background:linear-gradient(160deg,var(--brand) 0%,var(--brand-deep) 100%);
          border:none; box-shadow:0 8px 24px rgba(34,107,95,0.16); }}
        .fz-card.primary .ico {{ background:rgba(255,255,255,0.18); color:#fff; }}
        .fz-card.primary h3 {{ color:#fff; font-size:20px; }}
        .fz-card.primary .body {{ color:rgba(255,255,255,0.88); }}
        .fz-card.primary .tag {{ font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
          color:rgba(255,255,255,0.75); margin-bottom:10px; }}
        .fz-card.primary .go {{ color:#fff; }}
        .fz-card.primary:hover {{ box-shadow:0 12px 30px rgba(34,107,95,0.28); }}

        /* ---------- Scope footer ---------- */
        .fz-scope {{ background:var(--mist); border-radius:14px; padding:22px 28px; }}
        .fz-scope h4 {{ font-size:13px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase;
          color:var(--faint); margin:0 0 14px 0; }}
        .fz-scope .item {{ display:flex; gap:10px; align-items:flex-start; padding:6px 0; font-size:13.5px; color:var(--muted); }}
        .fz-scope .item b {{ color:var(--ink); font-weight:600; }}
        .fz-scope .item .ck {{ color:var(--brand); flex-shrink:0; margin-top:1px; }}

        /* ---------- Load gate ---------- */
        .fz-gate {{ max-width:520px; margin:6vh auto 0; text-align:center;
          background:var(--surface); border:1px solid var(--line); border-radius:18px;
          padding:48px 40px; box-shadow:0 8px 24px rgba(31,42,55,0.06); }}
        .fz-gate .logo {{ width:64px; height:64px; border-radius:16px; background:var(--brand-tint);
          color:var(--brand); display:flex; align-items:center; justify-content:center; margin:0 auto 20px; }}
        .fz-gate .eyebrow {{ font-size:12px; font-weight:600; letter-spacing:0.10em; text-transform:uppercase;
          color:var(--brand); margin-bottom:8px; }}
        .fz-gate h1 {{ font-size:30px; font-weight:800; margin:0 0 12px; }}
        .fz-gate p {{ font-size:15px; color:var(--muted); margin:0 auto 8px; max-width:380px; }}

        /* ---------- Buttons ---------- */
        .stButton > button {{ font-family:'Inter',sans-serif; font-weight:600; font-size:15px;
          border-radius:10px; padding:10px 22px; border:1px solid var(--brand); background:var(--brand); color:#fff;
          transition:background 160ms ease, box-shadow 160ms ease; cursor:pointer; }}
        .stButton > button:hover {{ background:#287a6d; box-shadow:0 2px 8px rgba(47,143,127,0.30); }}
        .stButton > button:focus-visible {{ outline:3px solid rgba(47,143,127,0.40); outline-offset:2px; }}

        [data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--line);
          border-radius:12px; padding:16px 20px; box-shadow:0 2px 8px rgba(31,42,55,0.04); }}
        [data-testid="stMetricLabel"] {{ color:var(--muted); }}

        @media (prefers-reduced-motion: reduce) {{ * {{ transition:none!important; animation:none!important; }} }}
        @media (max-width:880px) {{ .fz-grid,.fz-grid2 {{ grid-template-columns:1fr; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------
# Sidebar branding + status
# ---------------------------------------------------------------
def sidebar_brand() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="fz-brand">
              <div class="mark">{icon('shield_sm')}</div>
              <div>
                <div class="name">FraudZilla</div>
                <div class="sub">Fraud Risk Console</div>
              </div>
            </div>
            <div class="fz-side-divider"></div>
            """,
            unsafe_allow_html=True,
        )


def sidebar_status(loaded: bool) -> None:
    led = "on" if loaded else "off"
    label = "Models ready" if loaded else "Engine on standby"
    with st.sidebar:
        st.markdown(
            f'<div class="fz-side-status"><span class="led {led}"></span>{label}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------
# Overview building blocks
# ---------------------------------------------------------------
def status_bar(loaded: bool, version: str, entity_key: str, n_entities: int) -> None:
    pill = (
        f'<span class="fz-pill ok">{icon("check")}Models ready</span>'
        if loaded
        else '<span class="fz-pill wait">Engine on standby</span>'
    )
    st.markdown(
        f"""
        <div class="fz-statusbar">
          {pill}
          <div class="meta">
            <span>Entities <b>{n_entities}</b></span><span class="sep"></span>
            <span>Model <b>{version}</b></span><span class="sep"></span>
            <span>Entity key <b>{entity_key}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, subtitle_html: str) -> None:
    st.markdown(
        f"""
        <div class="fz-hero">
          <div class="fz-hero-row">
            <div class="fz-logo">{icon('shield')}</div>
            <div><div class="fz-eyebrow">{eyebrow}</div><h1>{title}</h1></div>
          </div>
          <p>{subtitle_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, hint: str = "") -> None:
    h = f'<span class="hint">{hint}</span>' if hint else ""
    st.markdown(f'<div class="fz-section"><h2>{title}</h2>{h}</div>', unsafe_allow_html=True)


def start_cards() -> None:
    """Informational workspace cards (navigation is via the sidebar)."""
    st.markdown(
        f"""
        <div class="fz-grid">
          <div class="fz-card primary">
            <div class="ico">{icon('receipt')}</div>
            <div class="tag">Most used</div>
            <h3>Transaction Scoring</h3>
            <div class="body">Score a single transaction and read its risk tier, recommended action, anomaly flag and escalation reason in one view.</div>
            <div class="go">Open from the sidebar {icon('arrow')}</div>
          </div>
          <div class="fz-card">
            <div class="ico">{icon('user')}</div>
            <h3>Entity Profile</h3>
            <div class="body">Track one entity's risk history and standing status as its transactions build up.</div>
            <div class="go">Entity Profile {icon('arrow')}</div>
          </div>
          <div class="fz-card">
            <div class="ico">{icon('chart')}</div>
            <h3>System Monitor</h3>
            <div class="body">Batch-score a CSV, review tier and action distributions, then export results.</div>
            <div class="go">System Monitor {icon('arrow')}</div>
          </div>
        </div>
        <div class="fz-grid2">
          <div class="fz-card">
            <div class="ico">{icon('search')}</div>
            <h3>Explanation Panel</h3>
            <div class="body">Surface the top SHAP drivers behind the CatBoost fraud probability for any transaction you've scored.</div>
            <div class="go">Explanation Panel {icon('arrow')}</div>
          </div>
          <div class="fz-card" style="justify-content:center;background:var(--brand-tint);border:1px dashed #BFE0D8;box-shadow:none;">
            <div style="font-size:13px;color:var(--brand-deep);font-weight:600;margin-bottom:6px;">Tip</div>
            <div class="body" style="color:var(--brand-deep);">Score a transaction or run a batch to start building entity profiles this session.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scope_list(items: list) -> None:
    rows = "".join(
        f'<div class="item"><span class="ck">{icon("check")}</span><span>{it}</span></div>'
        for it in items
    )
    st.markdown(
        f'<div class="fz-scope"><h4>How to read the scores</h4>{rows}</div>',
        unsafe_allow_html=True,
    )