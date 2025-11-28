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

# --- 2. APPLE UI DESIGN SYSTEM (CSS) ---
st.markdown("""
<style>
    /* GLOBAL FONTS & COLORS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #000000;
        color: #ffffff;
    }

    /* --- MAIN DASHBOARD LAYOUT --- */
    /* Constrain the main dashboard to a readable center column */
    .block-container {
        max-width: 1000px !important;
        margin: 0 auto !important;
        padding-top: 2rem !important;
    }

    /* --- SIDEBAR (THE CHAT DRAWER) --- */
    section[data-testid="stSidebar"] {
        width: 400px !important; /* Wider sidebar for better chat experience */
        background-color: #111111 !important; /* Slightly lighter black */
        border-right: 1px solid #222;
    }

    /* --- CHAT BUBBLES (SIDEBAR OPTIMIZED) --- */
    .chat-row { display: flex; margin-bottom: 15px; width: 100%; }
    .user-row { justify-content: flex-end; }
    .bot-row { justify-content: flex-start; }

    .chat-bubble {
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 85%; /* Wider bubbles since sidebar is narrow */
        font-size: 14px; /* Slightly smaller text for sidebar */
        line-height: 1.4;
        position: relative;
        word-wrap: break-word;
    }

    .user-bubble {
        background-color: #007AFF;
        color: white;
        border-bottom-right-radius: 2px;
    }

    .bot-bubble {
        background-color: #252525;
        color: #E5E5EA;
        border: 1px solid #333;
        border-bottom-left-radius: 2px;
    }
    
    .avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 8px;
        font-size: 14px;
    }
    .user-avatar { background-color: #222; order: 2; }
    .bot-avatar { background-color: #333; order: 0; }

    /* METRICS & CARDS */
    div[data-testid="stMetric"] {
        background-color: #1C1C1E;
        border: 1px solid #2C2C2E;
        padding: 15px;
        border-radius: 16px;
    }
    
    /* CLEANUP */
    header[data-testid="stHeader"] { background-color: transparent; }

</style>
""", unsafe_allow_html=True)

# --- 3. MEMORY FUNCTIONS ---
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

# --- 4. DATA FETCHING ---
def fetch_shopify_daily(domain, token):
    try:
        last_30 = (datetime.now() - timedelta(days=30)).isoformat()
        url = f"https://{domain}/admin/api/2023-10/orders.json?status=any&created_at_min={last_30}&limit=250"
        headers = {"X-Shopify-Access-Token": token}
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            orders = res.json().get('orders', [])
            data = [{"date": o['created_at'][:10], "sales": float(o['total_price'])} for o in orders]
            if not data: return pd.DataFrame(columns=["date", "sales"]), 0, None
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            daily_sales = df.groupby('date')['sales'].sum().reset_index()
            return daily_sales, df['sales'].sum(), None
        else: return None, 0, f"Shopify Error {res.status_code}: {res.text}"
    except Exception as e: return None, 0, f"Shopify Crash: {e}"

def fetch_meta_campaigns(token, account_id):
    try:
        url = f"https://graph.facebook.com/v17.0/act_{account_id}/insights"
        params = {'access_token': token, 'date_preset': 'last_30d', 'level': 'campaign', 'fields': 'campaign_name,spend,clicks,impressions,actions,action_values'}
        res = requests.get(url, params=params)
        if res.status_code == 200:
            data = res.json().get('data', [])
            if not data: return pd.DataFrame(columns=["Campaign", "Spend", "Sales", "ROAS", "Clicks"]), 0, None
            campaigns = []
            total_spend = 0
            for c in data:
                spend = float(c.get('spend', 0))
                total_spend += spend
                actions = c.get('action_values', [])
                sales_val = sum([float(a['value']) for a in actions if a['action_type'] == 'purchase']) if actions else 0
                campaigns.append({"Campaign": c.get('campaign_name'), "Spend": spend, "Sales": sales_val, "ROAS": round(sales_val/spend, 2) if spend>0 else 0, "Clicks": int(c.get('clicks', 0))})
            return pd.DataFrame(campaigns), total_spend, None
        else: return None, 0, f"Meta Error {res.status_code}: {res.text}"
    except Exception as e: return None, 0, f"Meta Crash: {e}"

# --- 5. APP LOGIC ---

if 'messages' not in st.session_state: st.session_state.messages = load_memory()
if 'logs' not in st.session_state: st.session_state.logs = []

# ==========================================
# ⬅️ SIDEBAR: THE AI CHAT DRAWER
# ==========================================
with st.sidebar:
    st.markdown("### AI Strategist")
    
    # 1. SETTINGS DRAWER (Collapsible)
    with st.expander("⚙️ Data Connection"):
        if st.button("🔄 Sync Data", type="primary", use_container_width=True):
            with st.spinner("Syncing..."):
                st.session_state.logs = [] 
                try:
                    s_domain, s_token = st.secrets["SHOPIFY_DOMAIN"], st.secrets["SHOPIFY_TOKEN"]
                    m_token, m_id = st.secrets["META_TOKEN"], st.secrets["META_ACCOUNT_ID"]
                    
                    daily_df, total_sales, s_err = fetch_shopify_daily(s_domain, s_token)
                    campaign_df, total_spend, m_err = fetch_meta_campaigns(m_token, m_id)
                    
                    if s_err: st.session_state.logs.append(s_err)
                    if m_err: st.session_state.logs.append(m_err)

                    if daily_df is not None and campaign_df is not None:
                        st.session_state['context'] = {
                            "daily_sales": daily_df, "campaigns": campaign_df,
                            "total_sales": total_sales, "total_spend": total_spend,
                            "roas": total_sales/total_spend if total_spend>0 else 0
                        }
                        if not st.session_state.logs: st.toast("Sync Complete", icon="✅")
                    else: st.toast("Sync Failed", icon="⚠️")
                except Exception as e: st.session_state.logs.append(f"Config Error: {e}")
        
        if st.session_state.logs:
            st.error(f"{len(st.session_state.logs)} Errors Found")
            for err in st.session_state.logs: st.caption(err)
            
        if st.button("Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # 2. CHAT HISTORY (Scrollable in sidebar)
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-row user-row">
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-row bot-row">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # 3. CHAT INPUT (Natively sticky in Sidebar)
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_memory("user", prompt)
        
        # Trick to show user message immediately
        st.markdown(f"""
        <div class="chat-row user-row">
            <div class="chat-bubble user-bubble">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            
            with st.status("Thinking...", expanded=False) as status:
                context_str = ""
                if 'context' in st.session_state:
                    cmp_sum = st.session_state['context']['campaigns'].to_string(index=False)
                    context_str = f"DATA:\nSales: {st.session_state['context']['total_sales']}\n\nCAMPAIGNS:\n{cmp_sum}"
                status.update(label="Complete", state="complete", expanded=False)
            
            history = st.session_state.messages[-30:] if len(st.session_state.messages) > 30 else st.session_state.messages
            final_prompt = f"You are an elite Media Buyer. Be concise, tactical, and data-driven.\n\n{context_str}"
            
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": final_prompt}] + [{"role": m["role"], "content": m["content"]} for m in history],
                stream=True
            )
            
            response_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            save_memory("assistant", response_text)
            st.rerun()
            
        except Exception as e: st.error(f"Error: {e}")


# ==========================================
# 📊 MAIN PAGE: THE DASHBOARD
# ==========================================

st.markdown("## Overview")

if 'context' in st.session_state:
    ctx = st.session_state['context']
    
    # 1. Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue (30d)", f"${ctx['total_sales']:,.0f}")
    c2.metric("Ad Spend", f"${ctx['total_spend']:,.0f}")
    c3.metric("ROAS", f"{ctx['roas']:.2f}x", delta="Target: 3.0x")
    c4.metric("Est. Profit", f"${(ctx['total_sales']*0.6 - ctx['total_spend']):,.0f}")
    
    st.markdown("<br>", unsafe_allow_html=True) 
    
    # 2. Chart
    st.subheader("Sales Velocity")
    if not ctx['daily_sales'].empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ctx['daily_sales']['date'], y=ctx['daily_sales']['sales'], marker_color='#007AFF', marker_line_width=0, opacity=0.9))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=300, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#333'))
        st.plotly_chart(fig, use_container_width=True)

    # 3. Table
    st.subheader("Active Campaigns")
    if not ctx['campaigns'].empty:
        st.dataframe(
            ctx['campaigns'].sort_values("Spend", ascending=False),
            column_config={
                "Spend": st.column_config.NumberColumn(format="$%.0f"),
                "Sales": st.column_config.NumberColumn(format="$%.0f"),
                "ROAS": st.column_config.NumberColumn(format="%.2fx"),
                "Clicks": st.column_config.NumberColumn(format="%d")
            },
            hide_index=True,
            use_container_width=True
        )
else:
    # Empty State
    st.info("👈 Open the Sidebar Settings to Sync your Data.")
