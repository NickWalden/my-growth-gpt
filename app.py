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
    /* 1. Global Typography & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #000000; /* True Black Background */
        color: #ffffff;
    }

    /* 2. STICKY TABS & HEADER */
    /* This makes the Tab Bar stick to the top of the screen */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 3rem; /* Height of the top decoration */
        z-index: 10;
        background-color: #000000;
        padding-top: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1C1C1E;
    }

    /* 3. iMESSAGE STYLE CHAT */
    /* User Message (Right Aligned, Blue) */
    div[data-testid="stChatMessage"]:nth-child(even) {
        flex-direction: row-reverse;
        text-align: right;
    }
    div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background-color: #007AFF !important; /* iOS Blue */
        color: white !important;
        border-radius: 20px 20px 4px 20px !important; /* Speech Bubble Shape */
        margin-right: 10px;
    }

    /* Assistant Message (Left Aligned, Grey) */
    div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background-color: #1C1C1E !important; /* iOS Dark Grey */
        border: 1px solid #2C2C2E;
        border-radius: 20px 20px 20px 4px !important;
        margin-left: 10px;
    }

    /* Avatars */
    div[data-testid="chatAvatarIcon-user"] {
        background-color: #1C1C1E !important;
    }
    div[data-testid="chatAvatarIcon-assistant"] {
        background-color: #007AFF !important;
    }

    /* 4. METRICS & CARDS */
    div[data-testid="stMetric"] {
        background-color: #1C1C1E;
        border: 1px solid #2C2C2E;
        padding: 15px;
        border-radius: 16px;
    }
    
    /* 5. SIDEBAR VISIBILITY FIX */
    /* We hide the red decoration bar but KEEP the hamburger/arrow accessible */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    
    /* 6. BOTTOM SPACING FOR STICKY INPUT */
    /* Ensures last message isn't hidden behind input box */
    .block-container {
        padding-bottom: 150px;
    }

</style>
""", unsafe_allow_html=True)

# --- 3. MEMORY FUNCTIONS ---
def load_memory():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="ChatHistory", usecols=[0, 1, 2], ttl=0)
        if df.empty or "role" not in df.columns:
            return []
        return df.to_dict("records")
    except Exception:
        return []

def save_memory(role, content):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        try:
            existing_data = conn.read(worksheet="ChatHistory", usecols=[0, 1, 2], ttl=0)
        except Exception:
            existing_data = pd.DataFrame(columns=["timestamp", "role", "content"])

        new_row = pd.DataFrame([{"timestamp": datetime.now().isoformat(), "role": role, "content": content}])
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        conn.update(worksheet="ChatHistory", data=updated_data)
    except Exception:
        pass

# --- 4. SILENT DATA FETCHING ---
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

# --- 5. MAIN APPLICATION UI ---

if 'messages' not in st.session_state: st.session_state.messages = load_memory()
if 'logs' not in st.session_state: st.session_state.logs = []

# SIDEBAR (Minimalist)
with st.sidebar:
    st.markdown("### Growth OS")
    
    # Primary Refresh Action
    if st.button("🔄 Sync Data", type="primary"):
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
                else: st.toast("Sync Failed. Check Logs.", icon="⚠️")
            except Exception as e: st.session_state.logs.append(f"Config Error: {e}")

    # Silent Log Drawer
    st.markdown("---")
    log_count = len(st.session_state.logs)
    if log_count > 0:
        with st.expander(f"⚠️ System Logs ({log_count})"):
            for err in st.session_state.logs: st.error(err)
    else:
        st.caption("✅ Systems Operational")
        
    st.markdown("---")
    if st.button("Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# MAIN CONTENT AREA
# We removed the top header text to save space for sticky tabs

# TAB NAVIGATION
tab1, tab2 = st.tabs(["Overview", "AI Strategist"])

# DASHBOARD TAB
with tab1:
    if 'context' in st.session_state:
        ctx = st.session_state['context']
        
        # 1. Metrics Cards
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
            fig.add_trace(go.Bar(
                x=ctx['daily_sales']['date'], y=ctx['daily_sales']['sales'],
                marker_color='#007AFF', marker_line_width=0, opacity=0.9
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), height=300,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#333')
            )
            st.plotly_chart(fig, use_container_width=True)

        # 3. Campaign Table
        st.subheader("Active Campaigns")
        if not ctx['campaigns'].empty:
            st.dataframe(
                ctx['campaigns'].sort_values("Spend", ascending=False),
                column_config={
                    "Spend": st.column_config.NumberColumn(format="$%.0f"),
                    "Sales": st.column_config.NumberColumn(format="$%.0f"),
                    "ROAS": st.column_config.NumberColumn(format="%.2fx"),
                    "Clicks": st.column_config.NumberColumn(format="%d"),
                },
                hide_index=True, use_container_width=True
            )
    else:
        st.info("Tap 'Sync Data' to initialize dashboard.")

# CHAT TAB
with tab2:
    # Sticky container for chat history
    chat_container = st.container()
    
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            # We iterate to apply CSS classes cleanly
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input Area - Stays sticky to bottom automatically
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Immediate user feedback
        with st.chat_message("user"): st.markdown(prompt)
        save_memory("user", prompt)
        
        try:
            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            
            with st.status("Thinking...", expanded=True) as status:
                st.write("Accessing Secure Context...")
                context_str = ""
                if 'context' in st.session_state:
                    cmp_sum = st.session_state['context']['campaigns'].to_string(index=False)
                    context_str = f"DATA:\nSales: {st.session_state['context']['total_sales']}\n\nCAMPAIGNS:\n{cmp_sum}"
                st.write("Processing Strategy...")
                status.update(label="Complete", state="complete", expanded=False)
            
            history = st.session_state.messages[-30:] if len(st.session_state.messages) > 30 else st.session_state.messages
            final_prompt = f"You are an elite Media Buyer. Be concise, tactical, and data-driven.\n\n{context_str}"
            
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": final_prompt}] + 
                         [{"role": m["role"], "content": m["content"]} for m in history],
                stream=True
            )
            with st.chat_message("assistant"): response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            save_memory("assistant", response)
            
            # Force refresh to ensure alignment applies to new messages
            st.rerun()
            
        except Exception as e: st.error(f"Error: {e}")
