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

# --- 2. MEMORY FUNCTIONS ---
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
        except:
            existing_data = pd.DataFrame(columns=["timestamp", "role", "content"])
        
        new_row = pd.DataFrame([{"timestamp": datetime.now().isoformat(), "role": role, "content": content}])
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        conn.update(worksheet="ChatHistory", data=updated_data)
    except Exception:
        pass

# --- 3. ADVANCED DATA FETCHING (CORRECTED) ---

def fetch_shopify_daily(domain, token):
    """Fetches orders and groups them by day for charting."""
    try:
        # Get last 30 days
        last_30 = (datetime.now() - timedelta(days=30)).isoformat()
        url = f"https://{domain}/admin/api/2023-10/orders.json?status=any&created_at_min={last_30}&limit=250"
        headers = {"X-Shopify-Access-Token": token}
        res = requests.get(url, headers=headers)
        
        if res.status_code == 200:
            orders = res.json().get('orders', [])
            
            # Process into DataFrame
            data = []
            for o in orders:
                data.append({
                    "date": o['created_at'][:10], # Extract YYYY-MM-DD
                    "sales": float(o['total_price'])
                })
            
            if not data:
                # If no orders, return empty structure (not None) to prevent crashing
                return pd.DataFrame(columns=["date", "sales"]), 0

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            daily_sales = df.groupby('date')['sales'].sum().reset_index()
            total_sales = df['sales'].sum()
            
            return daily_sales, total_sales
        else:
            st.error(f"Shopify Error {res.status_code}: {res.text}")
            return None, 0
    except Exception as e:
        st.error(f"Shopify Connection Failed: {e}")
        return None, 0

def fetch_meta_campaigns(token, account_id):
    """Fetches Campaign-level performance."""
    try:
        # CORRECTED URL: Removed 'purchases' from fields, added 'actions'
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
                # Return empty dataframe if no ad spend found
                return pd.DataFrame(columns=["Campaign", "Spend", "Sales", "ROAS", "Clicks"]), 0
            
            for c in data:
                spend = float(c.get('spend', 0))
                total_spend += spend
                
                # Get purchase value safely
                actions = c.get('action_values', [])
                sales_val = 0
                if actions:
                    # Look for 'purchase' action type
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
            st.error(f"Meta Error {res.status_code}: {res.text}")
            return None, 0
    except Exception as e:
        st.error(f"Meta Connection Failed: {e}")
        return None, 0

# --- 4. MAIN APP LOGIC ---

if 'messages' not in st.session_state:
    st.session_state.messages = load_memory()

# Sidebar Setup
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
    st.title("Growth GPT")
    st.caption("Pro Media Buyer Edition")
    st.divider()
    
    if st.button("🔄 Refresh Data", type="primary"):
        with st.spinner("Connecting to APIs..."):
            try:
                # Load Secrets
                s_domain = st.secrets["SHOPIFY_DOMAIN"]
                s_token = st.secrets["SHOPIFY_TOKEN"]
                m_token = st.secrets["META_TOKEN"]
                m_id = st.secrets["META_ACCOUNT_ID"]
                
                # Fetch Data
                daily_df, total_sales = fetch_shopify_daily(s_domain, s_token)
                campaign_df, total_spend = fetch_meta_campaigns(m_token, m_id)
                
                # Check if fetch was successful (df is not None)
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
                    st.error("❌ One of the data sources failed. See error details above.")
                    
            except KeyError as e:
                st.error(f"Missing Secret: {e}")
            except Exception as e:
                st.error(f"Unexpected Error: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 5. THE DASHBOARD TAB ---

# Organize Layout into Tabs
tab1, tab2 = st.tabs(["📊 Performance Dashboard", "💬 AI Strategist"])

with tab1:
    if 'context' in st.session_state:
        ctx = st.session_state['context']
        
        # 1. Top Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales (30d)", f"${ctx['total_sales']:,.2f}", delta="Shopify")
        c2.metric("Ad Spend (30d)", f"${ctx['total_spend']:,.2f}", delta="Meta", delta_color="inverse")
        c3.metric("ROAS", f"{ctx['roas']:.2f}x", delta="Target: 3.0x", delta_color="normal" if ctx['roas'] > 3 else "off")
        
        # Calculate Blended CPA (Cost Per Acquisition) estimate
        c4.metric("Est. Profit", f"${(ctx['total_sales']*0.6 - ctx['total_spend']):,.2f}", delta="Assuming 60% Margin")

        st.divider()

        # 2. The Chart (Sales Trend)
        st.subheader("📈 Sales Trend")
        
        if not ctx['daily_sales'].empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ctx['daily_sales']['date'],
                y=ctx['daily_sales']['sales'],
                name='Shopify Sales',
                marker_color='#00CC96'
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No sales data found for this period.")

        # 3. Campaign Table
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
            st.warning("No campaign data found.")

    else:
        st.info("👈 Click 'Refresh Data' in the sidebar to load your dashboard.")

# --- 6. THE CHAT TAB ---

with tab2:
    st.subheader("Chat with your Data")
    
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Analyze my campaigns..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        save_memory("user", prompt)
        
        # AI Logic
        try:
            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            
            context_str = ""
            if 'context' in st.session_state:
                cmp_summary = st.session_state['context']['campaigns'].to_string(index=False)
                sales_summary = f"Total Sales: {st.session_state['context']['total_sales']}"
                context_str = f"LATEST DATA:\n{sales_summary}\n\nCAMPAIGN BREAKDOWN:\n{cmp_summary}"
            
            system_prompt = f"""
            You are a Senior Media Buyer. 
            Use the data below to answer the user. 
            Focus on ROAS and Spend
