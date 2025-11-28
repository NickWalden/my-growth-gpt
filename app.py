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

# --- 4. APP STATE & SIDEBAR ---

if 'messages' not in st.session_state: st.session_state.messages = load_memory()
if 'logs' not in st.session_state: st.session_state.logs = []

with st.sidebar:
    st.markdown("### ⚙️ Console Settings")
    
    chat_width_pct = st.slider("Chat Width", 20, 60, 30, 5, format="%d%%")
    font_size = st.slider("Text Size", 12, 24, 15, 1, format="%dpx")
    
    st.divider()
    
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
        with st.expander(f"⚠️ Logs ({len(st.session_state.logs)})"):
            for err in st.session_state.logs: st.error(err)
            
    st.divider()
    if st.button("Clear Memory", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. CSS ARCHITECTURE ---
st.markdown(f"""
<style>
    /* GLOBAL RESET */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #000000;
        color: #ffffff;
        height: 100vh;
        overflow: hidden !important; 
    }}
    
    header[data-testid="stHeader"] {{ display: none; }}

    /* --- THE SCROLL FIX --- */
    .block-container {{
        max-width: 100%;
        padding: 1rem 1rem 0 1rem;
        height: 100vh;
        overflow: hidden !important;
    }}
    
    /* --- STICKY INPUT POSITIONING --- */
    [data-testid="stChatInput"] {{
        position: fixed !important;
        bottom: 0 !important;
        right: 1.5rem !important;
        left: auto !important;
        width: {chat_width_pct-2}% !important;
        min-width: 300px;
        background-color: #111111 !important;
        z-index: 9999 !important;
        border-top: 1px solid #333;
        padding-top: 15px !important;
        padding-bottom: 25px !important;
    }}
    
    /* --- TEXT SIZING FIX (Targeting Everything) --- */
    .chat-bubble, .user-bubble, .bot-bubble, .chat-bubble p, .user-bubble p, .bot-bubble p {{
        font-size: {font_size}px !important; 
        line-height: 1.5;
    }}
    
    /* Bubble Structure */
    .chat-bubble {{
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 85%;
        position: relative;
        word-wrap: break-word;
        margin-bottom: 2px;
    }}

    /* Colors */
    .user-bubble {{ background-color: #0A84FF; color: white; border-bottom-right-radius: 2px; }}
    .bot-bubble {{ background-color: #262626; color: #E5E5EA; border: 1px solid #333; border-bottom-left-radius: 2px; }}
    
    /* UI Utilities */
    .chat-row {{ display: flex; margin-bottom: 12px; width: 100%; }}
    .user-row {{ justify-content: flex-end; }}
    .bot-row {{ justify-content: flex-start; }}
    div[data-testid="stMetric"] {{ background-color: #111; border: 1px solid #222; padding: 15px; border-radius: 12px; }}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 🖥️ SPLIT LAYOUT
# ==========================================

dash_col, chat_col = st.columns([100-chat_width_pct, chat_width_pct], gap="medium")

# ------------------------------------------
# 📊 LEFT: DASHBOARD
# ------------------------------------------
with dash_col:
    # INDEPENDENT SCROLL AREA
    with st.container(height=850, border=False):
        st.markdown("## Overview")
        
        if 'context' in st.session_state:
            ctx = st.session_state['context']
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"${ctx['total_sales']:,.0f}")
            c2.metric("Spend", f"${ctx['total_spend']:,.0f}")
            c3.metric("ROAS", f"{ctx['roas']:.2f}x")
            c4.metric("Profit", f"${(ctx['total_sales']*0.6 - ctx['total_spend']):,.0f}")
            
            st.markdown("---")
            
            # Chart
            st.subheader("Sales Trend")
            if not ctx['daily_sales'].empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=ctx['daily_sales']['date'], y=ctx['daily_sales']['sales'], marker_color='#0A84FF', marker_line_width=0, opacity=0.9))
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=350, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#333'))
                st.plotly_chart(fig, use_container_width=True)

            # Table
            st.subheader("Campaigns")
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
                st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        else:
            st.info("👈 Sync Data from the sidebar to begin.")

# ------------------------------------------
# 💬 RIGHT: CHAT CONSOLE
# ------------------------------------------
with chat_col:
    st.markdown("### AI Strategist")
    
    # INDEPENDENT SCROLL AREA
    with st.container(height=780, border=False):
        
        chat_html = '<div style="padding-bottom: 20px;">' 
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-row user-row">
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>
                """
            else:
                chat_html += f"""
                <div class="chat-row bot-row">
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>
                """
        chat_html += "</div>"
        
        st.markdown(chat_html, unsafe_allow_html=True)

# ------------------------------------------
# ⌨️ GLOBAL INPUT
# ------------------------------------------
if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_memory("user", prompt)
    
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        context_str = ""
        if 'context' in st.session_state:
            cmp_sum = st.session_state['context']['campaigns'].to_string(index=False)
            context_str = f"DATA:\nSales: {st.session_state['context']['total_sales']}\n\nCAMPAIGNS:\n{cmp_sum}"
        
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
