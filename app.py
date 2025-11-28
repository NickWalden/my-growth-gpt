import streamlit as st
import openai
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_gsheets import GSheetsConnection

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Growth OS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MEMORY FUNCTIONS ---
def load_memory():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="ChatHistory", usecols=[0, 1, 2], ttl=0)
        if df.empty or "role" not in df.columns: return []
        return df.to_dict("records")
    except Exception: return []

def save_memory(role, content):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        try: existing_data = conn.read(worksheet="ChatHistory", usecols=[0, 1, 2], ttl=0)
        except Exception: existing_data = pd.DataFrame(columns=["timestamp", "role", "content"])
        new_row = pd.DataFrame([{"timestamp": datetime.now().isoformat(), "role": role, "content": content}])
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        conn.update(worksheet="ChatHistory", data=updated_data)
    except Exception: pass

# --- 3. DATA FETCHING ---
def fetch_shopify_data(domain, token):
    try:
        last_30 = (datetime.now() - timedelta(days=30)).isoformat()
        url = f"https://{domain}/admin/api/2023-10/orders.json?status=any&created_at_min={last_30}&limit=250"
        headers = {"X-Shopify-Access-Token": token}
        res = requests.get(url, headers=headers)
        
        if res.status_code == 200:
            orders = res.json().get('orders', [])
            daily_data = {}
            product_sales = {}
            total_revenue = 0
            
            for o in orders:
                date = o['created_at'][:10]
                val = float(o['total_price'])
                daily_data[date] = daily_data.get(date, 0) + val
                total_revenue += val
                for item in o.get('line_items', []):
                    p_name = item['title']
                    p_rev = float(item['price']) * item['quantity']
                    product_sales[p_name] = product_sales.get(p_name, 0) + p_rev

            df_daily = pd.DataFrame(list(daily_data.items()), columns=['date', 'sales'])
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            sorted_products = sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "daily_df": df_daily,
                "total_sales": total_revenue,
                "top_products": sorted_products,
                "order_count": len(orders),
                "aov": total_revenue / len(orders) if len(orders) > 0 else 0
            }, None
        else: return None, f"Shopify Error {res.status_code}: {res.text}"
    except Exception as e: return None, f"Shopify Crash: {e}"

def fetch_meta_data(token, account_id):
    try:
        base_url = f"https://graph.facebook.com/v17.0/act_{account_id}/insights"
        cmp_params = {'access_token': token, 'date_preset': 'last_30d', 'level': 'campaign', 'fields': 'campaign_name,spend,clicks,impressions,actions,action_values,cpm,ctr,cpc'}
        cmp_res = requests.get(base_url, params=cmp_params)
        ad_params = {'access_token': token, 'date_preset': 'last_7d', 'level': 'ad', 'fields': 'ad_name,spend,ctr,cpm,action_values', 'limit': 20}
        ad_res = requests.get(base_url, params=ad_params)

        if cmp_res.status_code == 200 and ad_res.status_code == 200:
            cmp_data = cmp_res.json().get('data', [])
            ad_data = ad_res.json().get('data', [])
            campaigns = []
            total_spend = 0
            for c in cmp_data:
                spend = float(c.get('spend', 0))
                total_spend += spend
                actions = c.get('action_values', [])
                sales_val = sum([float(a['value']) for a in actions if a['action_type'] == 'purchase']) if actions else 0
                campaigns.append({"Campaign": c.get('campaign_name'), "Spend": spend, "Sales": sales_val, "ROAS": round(sales_val/spend, 2) if spend>0 else 0, "CTR": float(c.get('ctr', 0)), "CPM": float(c.get('cpm', 0))})
            
            top_ads = []
            for a in ad_data:
                spend = float(a.get('spend', 0))
                if spend > 0:
                    actions = a.get('action_values', [])
                    sales_val = sum([float(act['value']) for act in actions if act['action_type'] == 'purchase']) if actions else 0
                    roas = round(sales_val/spend, 2)
                    top_ads.append(f"{a['ad_name']} | Spend:${spend:.0f} | ROAS:{roas}x | CTR:{a.get('ctr')}%")

            return {"campaign_df": pd.DataFrame(campaigns), "total_spend": total_spend, "top_ads_list": top_ads}, None
        else: return None, f"Meta Error: {cmp_res.text}"
    except Exception as e: return None, f"Meta Crash: {e}"

# --- 4. APP STATE ---
if 'messages' not in st.session_state: st.session_state.messages = load_memory()
if 'logs' not in st.session_state: st.session_state.logs = []

# --- 5. SIDEBAR SETTINGS ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    chat_width_pct = st.slider("Chat Width", 20, 60, 35, 5, format="%d%%")
    font_size = st.slider("Text Size", 12, 24, 14, 1, format="%dpx")
    st.divider()
    
    if st.button("🔄 Sync Data", type="primary", use_container_width=True):
        with st.spinner("Syncing..."):
            st.session_state.logs = [] 
            try:
                s_domain, s_token = st.secrets["SHOPIFY_DOMAIN"], st.secrets["SHOPIFY_TOKEN"]
                m_token, m_id = st.secrets["META_TOKEN"], st.secrets["META_ACCOUNT_ID"]
                shop_data, s_err = fetch_shopify_data(s_domain, s_token)
                meta_data, m_err = fetch_meta_data(m_token, m_id)
                if s_err: st.session_state.logs.append(s_err)
                if m_err: st.session_state.logs.append(m_err)
                if shop_data and meta_data:
                    ts = shop_data['total_sales']
                    tsp = meta_data['total_spend']
                    st.session_state['context'] = {"shopify": shop_data, "meta": meta_data, "roas": ts / tsp if tsp > 0 else 0}
                    if not st.session_state.logs: st.toast("Sync Complete", icon="✅")
                else: st.toast("Sync Failed", icon="⚠️")
            except Exception as e: st.session_state.logs.append(f"Config Error: {e}")

    if st.session_state.logs:
        with st.expander(f"⚠️ Logs ({len(st.session_state.logs)})"):
            for err in st.session_state.logs: st.error(err)
    
    st.divider()
    if st.button("Clear Memory", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 6. CSS STYLING (FIXED HEADER) ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #000000;
        color: #ffffff;
        height: 100vh;
        overflow: hidden !important; 
    }}
    
    /* --- FIX: RESTORE HEADER BUT MAKE TRANSPARENT --- */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        z-index: 100; /* Ensure button is clickable */
    }}
    /* Hide the red decoration line */
    header[data-testid="stHeader"] .decoration {{ display: none; }}

    .block-container {{ max-width: 100%; padding: 3rem 1rem 0 1rem; height: 100vh; overflow: hidden !important; }}
    
    /* Columns & Scrolling */
    div[data-testid="column"] {{ height: 90vh; overflow-y: auto; overflow-x: hidden; display: block; }}
    div[data-testid="column"]:nth-of-type(2) > div {{ padding-bottom: 150px !important; }}

    /* Input Position */
    [data-testid="stChatInput"] {{
        position: fixed !important; bottom: 0 !important; right: 1.5rem !important; left: auto !important;
        width: {chat_width_pct-2}% !important; min-width: 300px;
        background-color: #111111 !important; z-index: 9999 !important;
        border-top: 1px solid #333; padding-top: 15px !important; padding-bottom: 25px !important;
    }}
    
    /* Text Sizing */
    .chat-bubble, .chat-bubble * {{ font-size: {font_size}px !important; line-height: 1.5; }}
    
    /* Bubbles */
    .chat-bubble {{ padding: 12px 16px; border-radius: 18px; max-width: 85%; position: relative; word-wrap: break-word; margin-bottom: 4px; display: inline-block; }}
    .user-bubble {{ background-color: #0A84FF; color: white; border-bottom-right-radius: 2px; }}
    .bot-bubble {{ background-color: #262626; color: #E5E5EA; border: 1px solid #333; border-bottom-left-radius: 2px; }}
    
    /* Utilities */
    .chat-row {{ display: flex; margin-bottom: 12px; width: 100%; }}
    .user-row {{ justify-content: flex-end; }}
    .bot-row {{ justify-content: flex-start; }}
    div[data-testid="stMetric"] {{ background-color: #111; border: 1px solid #222; padding: 15px; border-radius: 12px; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🖥️ MAIN LAYOUT
# ==========================================
dash_col, chat_col = st.columns([100-chat_width_pct, chat_width_pct], gap="medium")

# --- LEFT: DASHBOARD ---
with dash_col:
    with st.container(height=850, border=False):
        st.markdown("## Overview")
        if 'context' in st.session_state:
            ctx = st.session_state['context']
            s_data = ctx['shopify']
            m_data = ctx['meta']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue (30d)", f"${s_data['total_sales']:,.0f}")
            c2.metric("Spend (30d)", f"${m_data['total_spend']:,.0f}")
            c3.metric("ROAS", f"{ctx['roas']:.2f}x")
            c4.metric("AOV", f"${s_data['aov']:.2f}")

            st.markdown("---")
            
            c_chart, c_prod = st.columns([2, 1])
            with c_chart:
                st.subheader("Sales Trend")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=s_data['daily_df']['date'], y=s_data['daily_df']['sales'], marker_color='#0A84FF'))
                fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with c_prod:
                st.subheader("Top Products")
                for name, rev in s_data['top_products']:
                    st.markdown(f"**{name}**")
                    st.caption(f"${rev:,.0f} revenue")
                    st.progress(min(rev / (s_data['total_sales'] or 1), 1.0))

            st.subheader("Campaign Performance")
            st.dataframe(m_data['campaign_df'].sort_values("Spend", ascending=False), column_config={"Spend": st.column_config.NumberColumn(format="$%.0f"), "Sales": st.column_config.NumberColumn(format="$%.0f"), "ROAS": st.column_config.NumberColumn(format="%.2fx"), "CTR": st.column_config.NumberColumn(format="%.2f%%")}, hide_index=True, use_container_width=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)
        else:
            st.info("👈 Sync Data from the sidebar to begin.")

# --- RIGHT: CHAT ---
with chat_col:
    with st.container(height=780, border=False):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""<div class="chat-row user-row"><div class="chat-bubble user-bubble">{msg['content']}</div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="chat-row bot-row"><div class="chat-bubble bot-bubble">{msg['content']}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br><br><br>", unsafe_allow_html=True)

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_memory("user", prompt)
    
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        context_str = ""
        if 'context' in st.session_state:
            ctx = st.session_state['context']
            s_data, m_data = ctx['shopify'], ctx['meta']
            top_prods = "\n".join([f"- {n}: ${r:.0f}" for n, r in s_data['top_products']])
            top_ads = "\n".join(m_data['top_ads_list'])
            context_str = f"OVERVIEW:\nTotal Sales: ${s_data['total_sales']}\nTotal Spend: ${m_data['total_spend']}\nROAS: {ctx['roas']:.2f}x\nAOV: ${s_data['aov']:.2f}\n\nTOP PRODUCTS:\n{top_prods}\n\nRECENT ADS:\n{top_ads}\n\nCAMPAIGNS:\n{m_data['campaign_df'].to_string(index=False)}"
        
        history = st.session_state.messages[-30:] if len(st.session_state.messages) > 30 else st.session_state.messages
        final_prompt = f"You are a Senior Media Buyer. Use this data to answer:\n{context_str}"
        stream = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": final_prompt}] + [{"role": m["role"], "content": m["content"]} for m in history], stream=True)
        
        response_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        save_memory("assistant", response_text)
        st.rerun()
    except Exception as e: st.error(f"Error: {e}")
