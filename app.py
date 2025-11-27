import streamlit as st
import openai
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_gsheets import GSheetsConnection

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Growth GPT Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Pro" look
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stDataFrame {
        border: 1px solid #333;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MEMORY FUNCTIONS (SIMPLIFIED) ---
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
    # Simplified to avoid nesting errors
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # 1. Try to read existing data
        try:
            existing_data = conn.read(worksheet="ChatHistory", usecols=[0, 1, 2], ttl=0)
        except Exception:
            existing_data = pd.DataFrame(columns=["timestamp", "role", "content"])

        # 2. Create new row
        new_row = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content
        }])
        
        # 3. Combine and save
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        conn.update(worksheet="ChatHistory", data=updated_data)
        
    except Exception as e:
        # Silently fail on write errors to keep app running
        print(f"Save Error: {e}")

# --- 3. DATA FETCHING ---

def fetch_shopify_daily(domain, token):
    try:
        last_30 = (datetime.now() - timedelta(days=30)).isoformat()
        url = f"https://{domain}/admin/api/2023-10/orders.json?status=any&created_at_min={last_30}&limit=250"
        headers = {"X-Shopify-Access-Token": token}
        res = requests.get(url, headers=headers)
        
        if res.status_code == 200:
            orders = res.json().get('orders', [])
            data = []
            for o in orders:
                data.append({
                    "date": o['created_at'][:10],
                    "sales": float(o['total_price'])
                })
            
            if not data:
                return pd.DataFrame(columns=["date", "sales"]), 0

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            daily_sales = df.groupby('date')['sales'].sum().reset_index()
            total_sales = df['sales'].sum()
            return daily_sales, total_sales
        else:
            return None, 0
    except Exception:
        return None, 0

def fetch_meta_campaigns(token, account_id):
    try:
        url = f"https://graph.facebook.com/v17.0/act_{account_id}/insights"
        params = {
            'access_token': token,
            'date_preset': 'last_30d',
            'level': 'campaign',
            'fields': 'campaign_name,spend,clicks,impressions,actions,action_values'
        }
        res = requests.get(url, params=params)
        
        if res.status_code == 200:
            data = res.json().get('data', [])
            campaigns = []
            total_spend = 0
            
            if not data:
                return pd.DataFrame(columns=["Campaign", "Spend", "Sales", "ROAS", "Clicks"]), 0
            
            for c in data:
                spend = float(c.get('spend', 0))
                total_spend += spend
                
                actions = c.get('action_values', [])
                sales_val = 0
                if actions:
                    sales_val = sum([float(a['value']) for a in actions if a['action_type'] == 'purchase'])
                
                campaigns.append({
                    "Campaign": c.get('campaign_name'),
                    "Spend": spend,
                    "Sales": sales_val,
                    "ROAS": round(sales_val / spend, 2) if spend > 0 else 0,
                    "Clicks": int(c.get('clicks', 0))
                })
            
            return pd.DataFrame(campaigns), total_spend
        else:
            return None, 0
    except Exception:
        return None, 0

# --- 4. MAIN APPLICATION ---

if 'messages' not in st.session_state:
    st.session_state.messages = load_memory()

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
    st.title("Growth GPT")
    st.caption("Pro Media Buyer Edition")
    st.divider()
    
    if st.button("🔄 Refresh Data", type="primary"):
        with st.spinner("Connecting..."):
            try:
                s_domain = st.secrets["SHOPIFY_DOMAIN"]
                s_token = st.secrets["SHOPIFY_TOKEN"]
                m_token = st.secrets["META_TOKEN"]
                m_id = st.secrets["META_ACCOUNT_ID"]
                
                daily_df, total_sales = fetch_shopify_daily(s_domain, s_token)
                campaign_df, total_spend = fetch_meta_campaigns(m_token, m_id)
                
                if daily_df is not None and campaign_df is not None:
                    roas = total_sales / total_spend if total_spend > 0 else 0
                    st.session_state['context'] = {
                        "daily_sales": daily_df,
                        "campaigns": campaign_df,
                        "total_sales": total_sales,
                        "total_spend": total_spend,
                        "roas": roas
                    }
                    st.success("✅ Data Loaded!")
                else:
                    st.error("❌ Data fetch failed.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# MAIN TABS
tab1, tab2 = st.tabs(["📊 Performance Dashboard", "💬 AI Strategist"])

# TAB 1: DASHBOARD
with tab1:
    if 'context' in st.session_state:
        ctx = st.session_state['context']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales (30d)", f"${ctx['total_sales']:,.2f}")
        c2.metric("Ad Spend (30d)", f"${ctx['total_spend']:,.2f}")
        c3.metric("ROAS", f"{ctx['roas']:.2f}x")
        c4.metric("Est. Profit", f"${(ctx['total_sales']*0.6 - ctx['total_spend']):,.2f}")

        st.divider()
        st.subheader("📈 Sales Trend")
        if not ctx['daily_sales'].empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ctx['daily_sales']['date'],
                y=ctx['daily_sales']['sales'],
                marker_color='#00CC96'
            ))
            fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏆 Campaign Performance")
        if not ctx['campaigns'].empty:
            st.dataframe(
                ctx['campaigns'].sort_values("Spend", ascending=False),
                column_config={
                    "Spend": st.column_config.NumberColumn(format="$%.2f"),
                    "Sales": st.column_config.NumberColumn(format="$%.2f"),
                    "ROAS": st.column_config.NumberColumn(format="%.2fx"),
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("👈 Click 'Refresh Data' to start.")

# TAB 2: CHAT
with tab2:
    st.subheader("Chat with your Data")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your ads..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        save_memory("user", prompt)
        
        try:
            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            
            context_str = ""
            if 'context' in st.session_state:
                cmp_sum = st.session_state['context']['campaigns'].to_string(index=False)
                context_str = f"DATA:\nSales: {st.session_state['context']['total_sales']}\n\nCAMPAIGNS:\n{cmp_sum}"
            
            final_prompt = f"You are a media buyer. Analyze this:\n{context_str}"
            
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": final_prompt}] + 
                         [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]],
                stream=True
            )
            
            with st.chat_message("assistant"):
                response = st.write_stream(stream)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            save_memory("assistant", response)
            
        except Exception as e:
            st.error(f"AI Error: {e}")
