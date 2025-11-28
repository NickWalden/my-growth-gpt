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

# --- 3. DATA FETCHING (WITH REAL COGS) ---

def get_product_costs(domain, token, variant_ids):
    """
    Fetches the actual Cost per Item from Shopify Inventory.
    This requires 'read_inventory' scope.
    """
    if not variant_ids:
        return {}
        
    cost_map = {} # {variant_id: cost}
    
    try:
        # 1. We need to get InventoryItem IDs from Variants first
        # Chunking IDs to avoid URL limit (50 at a time)
        chunks = [variant_ids[i:i + 50] for i in range(0, len(variant_ids), 50)]
        
        inventory_item_ids = []
        variant_to_inventory = {} # {variant_id: inventory_item_id}

        headers = {"X-Shopify-Access-Token": token}

        for chunk in chunks:
            ids_str = ",".join(map(str, chunk))
            url = f"https://{domain}/admin/api/2023-10/variants.json?ids={ids_str}&fields=id,inventory_item_id"
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                vars = res.json().get('variants', [])
                for v in vars:
                    variant_to_inventory[v['id']] = v['inventory_item_id']
                    inventory_item_ids.append(v['inventory_item_id'])

        # 2. Now fetch Costs from Inventory Items
        if inventory_item_ids:
            inv_chunks = [inventory_item_ids[i:i + 50] for i in range(0, len(inventory_item_ids), 50)]
            for chunk in inv_chunks:
                ids_str = ",".join(map(str, chunk))
                url = f"https://{domain}/admin/api/2023-10/inventory_items.json?ids={ids_str}&fields=id,cost"
                res = requests.get(url, headers=headers)
                if res.status_code == 200:
                    items = res.json().get('inventory_items', [])
                    inv_cost_map = {item['id']: float(item['cost'] or 0) for item in items}
                    
                    # Map back to Variant ID
                    for vid, inv_id in variant_to_inventory.items():
                        if inv_id in inv_cost_map:
                            cost_map[vid] = inv_cost_map[inv_id]
                            
    except Exception as e:
        print(f"COGS Fetch Error: {e}")
        
    return cost_map

def fetch_shopify_data(domain, token, fallback_margin):
    try:
        last_30 = (datetime.now() - timedelta(days=30)).isoformat()
        url = f"https://{domain}/admin/api/2023-10/orders.json?status=any&created_at_min={last_30}&limit=250"
        headers = {"X-Shopify-Access-Token": token}
        res = requests.get(url, headers=headers)
        
        if res.status_code == 200:
            orders = res.json().get('orders', [])
            
            # Collect all Variant IDs to fetch costs
            all_variant_ids = set()
            for o in orders:
                for item in o.get('line_items', []):
                    if item.get('variant_id'):
                        all_variant_ids.add(item['variant_id'])
            
            # Fetch Real Costs
            real_costs = get_product_costs(domain, token, list(all_variant_ids))
            
            daily_map = {} # {date: {'sales': 0, 'cogs': 0}}
            product_sales = {}
            total_revenue = 0
            total_cogs = 0
            
            for o in orders:
                date = o['created_at'][:10]
                if date not in daily_map: daily_map[date] = {'sales': 0, 'cogs': 0}
                
                order_rev = float(o['total_price'])
                order_cogs = 0
                
                # Calculate COGS for this order
                for item in o.get('line_items', []):
                    vid = item.get('variant_id')
                    price = float(item['price'])
                    qty = item['quantity']
                    
                    # Use Real Cost if available, else use Fallback Margin
                    if vid in real_costs and real_costs[vid] > 0:
                        item_cost = real_costs[vid]
                    else:
                        # Fallback: Cost = Price * (1 - Margin)
                        item_cost = price * (1 - fallback_margin)
                    
                    order_cogs += (item_cost * qty)
                    
                    # Product Breakdown
                    p_name = item['title']
                    product_sales[p_name] = product_sales.get(p_name, 0) + (price * qty)

                daily_map[date]['sales'] += order_rev
                daily_map[date]['cogs'] += order_cogs
                total_revenue += order_rev
                total_cogs += order_cogs

            # Format Data
            daily_list = [{'date': k, 'sales': v['sales'], 'cogs': v['cogs']} for k, v in daily_map.items()]
            df_daily = pd.DataFrame(daily_list)
            if not df_daily.empty:
                df_daily['date'] = pd.to_datetime(df_daily['date'])
                df_daily = df_daily.sort_values('date')
            
            sorted_products = sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "daily_df": df_daily,
                "total_sales": total_revenue,
                "total_cogs": total_cogs,
                "top_products": sorted_products,
                "aov": total_revenue / len(orders) if len(orders) > 0 else 0
            }, None
        else: return None, f"Shopify Error {res.status_code}: {res.text}"
    except Exception as e: return None, f"Shopify Crash: {e}"

def fetch_meta_data(token, account_id):
    try:
        base_url = f"https://graph.facebook.com/v17.0/act_{account_id}/insights"
        
        # 1. Campaign Level
        cmp_params = {'access_token': token, 'date_preset': 'last_30d', 'level': 'campaign', 'fields': 'campaign_name,spend,clicks,impressions,actions,action_values,cpm,ctr,cpc'}
        cmp_res = requests.get(base_url, params=cmp_params)
        
        # 2. Daily Spend
        daily_params = {'access_token': token, 'date_preset': 'last_30d', 'level': 'account', 'time_increment': 1, 'fields': 'spend,date_start'}
        daily_res = requests.get(base_url, params=daily_params)
        
        # 3. Ad Level
        ad_params = {'access_token': token, 'date_preset': 'last_7d', 'level': 'ad', 'fields': 'ad_name,spend,ctr,cpm,action_values', 'limit': 20}
        ad_res = requests.get(base_url, params=ad_params)

        if cmp_res.status_code == 200 and daily_res.status_code == 200:
            cmp_data = cmp_res.json().get('data', [])
            daily_data = daily_res.json().get('data', [])
            ad_data = ad_res.json().get('data', [])
            
            daily_spend_map = {}
            total_spend = 0
            for d in daily_data:
                daily_spend_map[d['date_start']] = float(d['spend'])
                total_spend += float(d['spend'])
                
            df_daily_spend = pd.DataFrame(list(daily_spend_map.items()), columns=['date', 'spend'])
            df_daily_spend['date'] = pd.to_datetime(df_daily_spend['date'])

            campaigns = []
            for c in cmp_data:
                spend = float(c.get('spend', 0))
                actions = c.get('action_values', [])
                sales_val = sum([float(a['value']) for a in actions if a['action_type'] == 'purchase']) if actions else 0
                campaigns.append({"Campaign": c.get('campaign_name'), "Spend": spend, "Sales": sales_val, "ROAS": round(sales_val/spend, 2) if spend>0 else 0, "CTR": float(c.get('ctr', 0))})
            
            top_ads = []
            for a in ad_data:
                spend = float(a.get('spend', 0))
                if spend > 0:
                    actions = a.get('action_values', [])
                    sales_val = sum([float(act['value']) for act in actions if act['action_type'] == 'purchase']) if actions else 0
                    roas = round(sales_val/spend, 2)
                    top_ads.append(f"{a['ad_name']} | Spend:${spend:.0f} | ROAS:{roas}x | CTR:{a.get('ctr')}%")

            return {
                "campaign_df": pd.DataFrame(campaigns),
                "daily_spend_df": df_daily_spend,
                "total_spend": total_spend,
                "top_ads_list": top_ads
            }, None
        else: return None, f"Meta Error: {cmp_res.text}"
    except Exception as e: return None, f"Meta Crash: {e}"

# --- 4. APP STATE ---
if 'messages' not in st.session_state: st.session_state.messages = load_memory()
if 'logs' not in st.session_state: st.session_state.logs = []

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    chat_width_pct = st.slider("Chat Width", 20, 60, 35, 5, format="%d%%")
    font_size = st.slider("Text Size", 12, 24, 14, 1, format="%dpx")
    
    st.markdown("---")
    st.caption("Fallback Margin (If Shopify COGS missing)")
    margin_pct = st.slider("Margin %", 10, 90, 60, 5, format="%d%%") / 100.0
    
    st.divider()
    
    if st.button("🔄 Sync Data", type="primary", use_container_width=True):
        with st.spinner("Syncing..."):
            st.session_state.logs = [] 
            try:
                s_domain, s_token = st.secrets["SHOPIFY_DOMAIN"], st.secrets["SHOPIFY_TOKEN"]
                m_token, m_id = st.secrets["META_TOKEN"], st.secrets["META_ACCOUNT_ID"]
                
                shop_data, s_err = fetch_shopify_data(s_domain, s_token, margin_pct)
                meta_data, m_err = fetch_meta_data(m_token, m_id)
                
                if s_err: st.session_state.logs.append(s_err)
                if m_err: st.session_state.logs.append(m_err)

                if shop_data and meta_data:
                    # --- TRUE PROFIT CALCULATION ---
                    df_s = shop_data['daily_df']
                    df_m = meta_data['daily_spend_df']
                    
                    if not df_s.empty and not df_m.empty:
                        df_merged = pd.merge(df_s, df_m, on='date', how='outer').fillna(0)
                        
                        # Net Profit = Revenue - COGS - Ad Spend
                        df_merged['gross_profit'] = df_merged['sales'] - df_merged['cogs']
                        df_merged['net_profit'] = df_merged['gross_profit'] - df_merged['spend']
                        df_merged = df_merged.sort_values('date')
                        
                        total_net_profit = df_merged['net_profit'].sum()
                    else:
                        df_merged = pd.DataFrame()
                        total_net_profit = 0

                    st.session_state['context'] = {
                        "shopify": shop_data,
                        "meta": meta_data,
                        "profit_df": df_merged,
                        "total_net_profit": total_net_profit,
                        "roas": shop_data['total_sales'] / meta_data['total_spend'] if meta_data['total_spend'] > 0 else 0
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

# --- 6. CSS (HEADER FIX & SCROLL) ---
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
    
    /* --- FIX: HEADER VISIBLE BUT TRANSPARENT --- */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        z-index: 999;
    }}
    /* Hide colored strip */
    .stApp > header {{ background-color: transparent; }}
    
    .block-container {{
        max-width: 100%;
        padding: 4rem 1rem 0 1rem; /* Added top padding for header space */
        height: 100vh;
        overflow: hidden !important;
    }}
    
    div[data-testid="column"] {{
        height: 90vh;
        overflow-y: auto;
        overflow-x: hidden;
        display: block;
    }}
    div[data-testid="column"]:nth-of-type(2) > div {{ padding-bottom: 150px !important; }}

    [data-testid="stChatInput"] {{
        position: fixed !important; bottom: 0 !important; right: 1.5rem !important; left: auto !important;
        width: {chat_width_pct-2}% !important; min-width: 300px;
        background-color: #111111 !important; z-index: 9999 !important;
        border-top: 1px solid #333; padding-top: 15px !important; padding-bottom: 25px !important;
    }}
    
    .chat-bubble, .chat-bubble * {{ font-size: {font_size}px !important; line-height: 1.5; }}
    .chat-bubble {{ padding: 12px 16px; border-radius: 18px; max-width: 85%; position: relative; word-wrap: break-word; margin-bottom: 4px; display: inline-block; }}
    .user-bubble {{ background-color: #0A84FF; color: white; border-bottom-right-radius: 2px; }}
    .bot-bubble {{ background-color: #262626; color: #E5E5EA; border: 1px solid #333; border-bottom-left-radius: 2px; }}
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
            c1.metric("Revenue", f"${s_data['total_sales']:,.0f}")
            c2.metric("True Profit", f"${ctx['total_net_profit']:,.0f}", delta="Net (Sales - COGS - Ads)")
            c3.metric("ROAS", f"{ctx['roas']:.2f}x")
            
            # Determine Margin Source
            margin_source = "Real COGS" if s_data['total_cogs'] > 0 else f"Est. {margin_pct*100}%"
            c4.metric("COGS Source", margin_source)

            st.markdown("---")
            
            # --- PROFIT CHART ---
            st.subheader("Daily Net Profit")
            df = ctx['profit_df']
            if not df.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df['date'], y=df['net_profit'], name='Net Profit', marker_color=df['net_profit'].apply(lambda x: '#00C853' if x >= 0 else '#FF3D00')))
                fig.add_trace(go.Scatter(x=df['date'], y=df['spend'], name='Ad Spend', line=dict(color='#888888', width=2, dash='dot')))
                fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)

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
            
            top_ads = "\n".join(m_data['top_ads_list'])
            
            context_str = f"""
            OVERVIEW:
            - Net Profit: ${ctx['total_net_profit']:,.2f}
            - Revenue: ${s_data['total_sales']}
            - COGS: ${s_data['total_cogs']}
            - Ad Spend: ${m_data['total_spend']}
            - ROAS: {ctx['roas']:.2f}x
            
            RECENT TOP ADS:
            {top_ads}
            
            CAMPAIGNS:
            {m_data['campaign_df'].to_string(index=False)}
            """
        
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
