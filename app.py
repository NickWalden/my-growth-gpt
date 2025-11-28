import streamlit as st
import openai
import pandas as pd
import requests
import plotly.graph_objects as go
import json
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

def get_product_costs(domain, token, variant_ids):
    if not variant_ids: return {}
    cost_map = {}
    try:
        chunks = [variant_ids[i:i + 50] for i in range(0, len(variant_ids), 50)]
        headers = {"X-Shopify-Access-Token": token}
        for chunk in chunks:
            ids_str = ",".join(map(str, chunk))
            url = f"https://{domain}/admin/api/2023-10/variants.json?ids={ids_str}&fields=id,inventory_item_id"
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                vars = res.json().get('variants', [])
                inv_ids = [v['inventory_item_id'] for v in vars]
                var_map = {v['inventory_item_id']: v['id'] for v in vars}
                if inv_ids:
                    inv_str = ",".join(map(str, inv_ids))
                    url2 = f"https://{domain}/admin/api/2023-10/inventory_items.json?ids={inv_str}&fields=id,cost"
                    res2 = requests.get(url2, headers=headers)
                    if res2.status_code == 200:
                        items = res2.json().get('inventory_items', [])
                        for item in items:
                            vid = var_map.get(item['id'])
                            if vid: cost_map[vid] = float(item['cost'] or 0)
    except Exception: pass
    return cost_map

def fetch_shopify_data(domain, token, fallback_margin, start_date, end_date):
    try:
        start_iso = start_date.strftime('%Y-%m-%dT00:00:00')
        end_iso = end_date.strftime('%Y-%m-%dT23:59:59')
        url = f"https://{domain}/admin/api/2023-10/orders.json?status=any&created_at_min={start_iso}&created_at_max={end_iso}&limit=250"
        headers = {"X-Shopify-Access-Token": token}
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            orders = res.json().get('orders', [])
            all_vids = set()
            for o in orders:
                for i in o.get('line_items', []):
                    if i.get('variant_id'): all_vids.add(i['variant_id'])
            real_costs = get_product_costs(domain, token, list(all_vids))
            
            daily_map = {}
            prod_sales = {}
            total_rev = 0
            total_cogs = 0
            for o in orders:
                date = o['created_at'][:10]
                if date not in daily_map: daily_map[date] = {'sales': 0, 'cogs': 0}
                rev = float(o['total_price'])
                cogs = 0
                for i in o.get('line_items', []):
                    vid = i.get('variant_id')
                    price = float(i['price'])
                    qty = i['quantity']
                    cost = real_costs.get(vid, price * (1 - fallback_margin))
                    cogs += (cost * qty)
                    prod_sales[i['title']] = prod_sales.get(i['title'], 0) + (price * qty)
                daily_map[date]['sales'] += rev
                daily_map[date]['cogs'] += cogs
                total_rev += rev
                total_cogs += cogs

            df_daily = pd.DataFrame([{'date': k, 'sales': v['sales'], 'cogs': v['cogs']} for k, v in daily_map.items()])
            if not df_daily.empty:
                df_daily['date'] = pd.to_datetime(df_daily['date'])
                df_daily = df_daily.sort_values('date')
            return {
                "daily_df": df_daily, "total_sales": total_rev, "total_cogs": total_cogs,
                "top_products": sorted(prod_sales.items(), key=lambda x: x[1], reverse=True)[:5],
                "aov": total_rev / len(orders) if len(orders) > 0 else 0
            }, None
        else: return None, f"Shopify Error {res.status_code}"
    except Exception as e: return None, f"Shopify Crash: {e}"

def fetch_ad_creatives_batch(token, ad_ids):
    """
    Step 2: Fetch Creative Data using the Ad IDs.
    UPGRADE: We now request 'thumbnail_url' AND 'image_url' AND check nested fields.
    """
    if not ad_ids: return {}
    image_map = {}
    
    chunks = [ad_ids[i:i + 50] for i in range(0, len(ad_ids), 50)]
    
    for chunk in chunks:
        try:
            ids_str = ",".join(chunk)
            # UPGRADE: Requesting image_url explicitly first
            url = f"https://graph.facebook.com/v17.0/?ids={ids_str}&fields=creative{{image_url,thumbnail_url,object_story_spec,asset_feed_spec}}&access_token={token}"
            res = requests.get(url)
            
            if res.status_code == 200:
                data = res.json()
                for ad_id, val in data.items():
                    creative = val.get('creative', {})
                    img = None
                    
                    # 1. High Res Image (Standard Ads)
                    img = creative.get('image_url')
                    
                    # 2. Fallback to Thumbnail (Videos)
                    if not img: img = creative.get('thumbnail_url')
                    
                    # 3. Object Story (Posts/Videos)
                    if not img:
                        try:
                            spec = creative.get('object_story_spec', {})
                            # Check Full Picture first (Highest Res)
                            img = spec.get('link_data', {}).get('picture') or \
                                  spec.get('photo_data', {}).get('image_url') or \
                                  spec.get('video_data', {}).get('image_url')
                        except: pass
                    
                    # 4. Dynamic Creative (DCO)
                    if not img:
                        try:
                            images = creative.get('asset_feed_spec', {}).get('images', [])
                            if images: img = images[0].get('url')
                        except: pass
                    
                    if img:
                        image_map[ad_id] = img
        except Exception: pass
        
    return image_map

def fetch_meta_data(token, account_id, start_date, end_date):
    try:
        base_url = f"https://graph.facebook.com/v17.0/act_{account_id}/insights"
        time_range = json.dumps({'since': start_date.strftime('%Y-%m-%d'), 'until': end_date.strftime('%Y-%m-%d')})
        
        # 1. Campaign Level
        cmp_params = {'access_token': token, 'time_range': time_range, 'level': 'campaign', 'fields': 'campaign_name,spend,clicks,impressions,actions,action_values,cpm,ctr,cpc', 'limit': 100}
        cmp_res = requests.get(base_url, params=cmp_params)
        
        # 2. Daily Spend
        daily_params = {'access_token': token, 'time_range': time_range, 'level': 'account', 'time_increment': 1, 'fields': 'spend,date_start', 'limit': 100}
        daily_res = requests.get(base_url, params=daily_params)
        
        # 3. Ad Level
        ad_params = {
            'access_token': token, 
            'time_range': time_range, 
            'level': 'ad',
            'fields': 'ad_id,ad_name,spend,ctr,cpm,action_values', 
            'limit': 50, 
            'sort': ['spend_descending']
        }
        ad_res = requests.get(base_url, params=ad_params)

        if cmp_res.status_code != 200: return None, f"Meta Campaign Error: {cmp_res.text}"
        if daily_res.status_code != 200: return None, f"Meta Daily Error: {daily_res.text}"
        if ad_res.status_code != 200: return None, f"Meta Ad Error: {ad_res.text}"

        cmp_data = cmp_res.json().get('data', [])
        daily_data = daily_res.json().get('data', [])
        ad_data = ad_res.json().get('data', [])
        
        daily_spend = [{'date': d['date_start'], 'spend': float(d['spend'])} for d in daily_data]
        df_daily_spend = pd.DataFrame(daily_spend)
        if not df_daily_spend.empty: df_daily_spend['date'] = pd.to_datetime(df_daily_spend['date'])
        total_spend = sum([d['spend'] for d in daily_spend])

        campaigns = []
        for c in cmp_data:
            spend = float(c.get('spend', 0))
            actions = c.get('action_values', [])
            sales_val = sum([float(a['value']) for a in actions if a['action_type'] == 'purchase']) if actions else 0
            campaigns.append({"Campaign": c.get('campaign_name'), "Spend": spend, "Sales": sales_val, "ROAS": round(sales_val/spend, 2) if spend>0 else 0, "CTR": float(c.get('ctr', 0))})
        
        gallery_ads = []
        ad_ids_to_fetch = []
        
        for a in ad_data:
            spend = float(a.get('spend', 0))
            if spend > 0 or int(a.get('impressions', 0)) > 10:
                actions = a.get('action_values', [])
                sales_val = sum([float(act['value']) for act in actions if act['action_type'] == 'purchase']) if actions else 0
                ad_id = a.get('ad_id') or a.get('id')
                
                if ad_id:
                    ad_ids_to_fetch.append(ad_id)
                    gallery_ads.append({
                        "id": ad_id,
                        "name": a['ad_name'], "spend": spend, 
                        "roas": round(sales_val/spend, 2) if spend>0 else 0,
                        "ctr": float(a.get('ctr', 0)), "cpm": float(a.get('cpm', 0))
                    })
        
        if ad_ids_to_fetch:
            image_map = fetch_ad_creatives_batch(token, list(set(ad_ids_to_fetch)))
            for ad in gallery_ads:
                ad['image_url'] = image_map.get(ad['id'])

        return {
            "campaign_df": pd.DataFrame(campaigns), "daily_spend_df": df_daily_spend,
            "total_spend": total_spend, "gallery_ads": gallery_ads
        }, None

    except Exception as e: return None, f"Meta Crash: {e}"

# --- 4. APP STATE ---
if 'messages' not in st.session_state: st.session_state.messages = load_memory()
if 'logs' not in st.session_state: st.session_state.logs = []
if 'last_synced_dates' not in st.session_state: st.session_state.last_synced_dates = None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    preset = st.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "This Month", "Last Month", "Custom"], index=1)
    today = datetime.now().date()
    if preset == "Last 7 Days": s_d, e_d = today - timedelta(days=7), today
    elif preset == "Last 30 Days": s_d, e_d = today - timedelta(days=30), today
    elif preset == "This Month": s_d, e_d = today.replace(day=1), today
    elif preset == "Last Month": 
        first = today.replace(day=1); e_d = first - timedelta(days=1); s_d = e_d.replace(day=1)
    else: 
        cols = st.columns(2); s_d = cols[0].date_input("Start", value=today - timedelta(days=30)); e_d = cols[1].date_input("End", value=today)

    st.divider()
    chat_width_pct = st.slider("Chat Width", 20, 60, 35, 5, format="%d%%")
    font_size = st.slider("Text Size", 12, 24, 14, 1, format="%dpx")
    margin_pct = st.slider("Margin % (Fallback)", 10, 90, 60, 5, format="%d%%") / 100.0
    st.divider()

    def run_sync():
        with st.spinner("Syncing..."):
            st.session_state.logs = [] 
            try:
                s_domain, s_token = st.secrets["SHOPIFY_DOMAIN"], st.secrets["SHOPIFY_TOKEN"]
                m_token, m_id = st.secrets["META_TOKEN"], st.secrets["META_ACCOUNT_ID"]
                shop_data, s_err = fetch_shopify_data(s_domain, s_token, margin_pct, s_d, e_d)
                meta_data, m_err = fetch_meta_data(m_token, m_id, s_d, e_d)
                if s_err: st.session_state.logs.append(s_err)
                if m_err: st.session_state.logs.append(m_err)
                if shop_data and meta_data:
                    df_s, df_m = shop_data['daily_df'], meta_data['daily_spend_df']
                    if not df_s.empty and not df_m.empty:
                        df_merged = pd.merge(df_s, df_m, on='date', how='outer').fillna(0)
                        df_merged['gross_profit'] = df_merged['sales'] - df_merged['cogs']
                        df_merged['net_profit'] = df_merged['gross_profit'] - df_merged['spend']
                        df_merged = df_merged.sort_values('date')
                        total_net_profit = df_merged['net_profit'].sum()
                    else: df_merged = pd.DataFrame(); total_net_profit = 0
                    st.session_state['context'] = {
                        "shopify": shop_data, "meta": meta_data, "profit_df": df_merged,
                        "total_net_profit": total_net_profit, "date_range": f"{s_d} to {e_d}",
                        "roas": shop_data['total_sales'] / meta_data['total_spend'] if meta_data['total_spend'] > 0 else 0
                    }
                    st.session_state.last_synced_dates = (s_d, e_d)
                    if not st.session_state.logs: st.toast("Sync Complete", icon="✅")
                else: st.toast("Sync Failed", icon="⚠️")
            except Exception as e: st.session_state.logs.append(f"Config Error: {e}")

    if st.session_state.last_synced_dates != (s_d, e_d): run_sync()
    if st.button("🔄 Force Sync", type="secondary", use_container_width=True): run_sync()
    if st.session_state.logs:
        with st.expander(f"⚠️ Logs ({len(st.session_state.logs)})"):
            for err in st.session_state.logs: st.error(err)
    st.divider()
    if st.button("Clear Memory", type="secondary", use_container_width=True): st.session_state.messages = []; st.rerun()

# --- 6. CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: #000; color: #fff; height: 100vh; overflow: hidden !important; }}
    
    header[data-testid="stHeader"] {{ background-color: transparent !important; z-index: 999; }}
    .stApp > header {{ background-color: transparent; }}
    
    .block-container {{ max-width: 100%; padding: 4rem 1rem 0 1rem; height: 100vh; overflow: hidden !important; }}
    div[data-testid="column"] {{ height: 90vh; overflow-y: auto; overflow-x: hidden; display: block; }}
    div[data-testid="column"]:nth-of-type(2) > div {{ padding-bottom: 150px !important; }}
    [data-testid="stChatInput"] {{ position: fixed !important; bottom: 0 !important; right: 1.5rem !important; left: auto !important; width: {chat_width_pct-2}% !important; min-width: 300px; background-color: #111 !important; z-index: 9999 !important; border-top: 1px solid #333; padding-top: 15px !important; padding-bottom: 25px !important; }}
    .chat-bubble, .chat-bubble * {{ font-size: {font_size}px !important; line-height: 1.5; }}
    .chat-bubble {{ padding: 12px 16px; border-radius: 18px; max-width: 85%; position: relative; word-wrap: break-word; margin-bottom: 4px; display: inline-block; }}
    .user-bubble {{ background-color: #0A84FF; color: white; border-bottom-right-radius: 2px; }}
    .bot-bubble {{ background-color: #262626; color: #E5E5EA; border: 1px solid #333; border-bottom-left-radius: 2px; }}
    .chat-row {{ display: flex; margin-bottom: 12px; width: 100%; }}
    .user-row {{ justify-content: flex-end; }}
    .bot-row {{ justify-content: flex-start; }}
    div[data-testid="stMetric"] {{ background-color: #111; border: 1px solid #222; padding: 15px; border-radius: 12px; }}
    
    /* GALLERY STYLES */
    .ad-card {{ background-color: #111; border: 1px solid #222; border-radius: 12px; overflow: hidden; margin-bottom: 20px; transition: transform 0.2s; position: relative; aspect-ratio: 4/5; }}
    .ad-card:hover {{ border-color: #444; transform: translateY(-2px); z-index: 10; }}
    .ad-bg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-size: cover; background-position: center; filter: blur(10px) brightness(0.5); z-index: 1; }}
    /* Force Image to Fill Card with Cover */
    .ad-image {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover !important; z-index: 2; }}
    
    .ad-overlay {{ position: absolute; bottom: 0; left: 0; width: 100%; background: rgba(0,0,0,0.85); padding: 10px; transform: translateY(100%); transition: transform 0.2s ease; z-index: 3; border-top: 1px solid #333; }}
    .ad-card:hover .ad-overlay {{ transform: translateY(0); }}
    .ad-badge-top {{ position: absolute; top: 8px; right: 8px; z-index: 4; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; backdrop-filter: blur(4px); }}
    .text-sm {{ font-size: 11px; color: #aaa; margin-bottom: 2px; }}
    .text-val {{ font-size: 13px; font-weight: 600; color: #fff; }}
    .row-split {{ display: flex; justify-content: space-between; }}
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
            s_data, m_data = ctx['shopify'], ctx['meta']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"${s_data['total_sales']:,.0f}")
            c2.metric("True Profit", f"${ctx['total_net_profit']:,.0f}", delta="Net")
            c3.metric("ROAS", f"{ctx['roas']:.2f}x")
            c4.metric("COGS Source", "Real COGS" if s_data['total_cogs'] > 0 else f"Est. {margin_pct*100}%")
            st.markdown("---")
            
            tab1, tab2, tab3 = st.tabs(["Profit Chart", "Creative Gallery", "Campaigns"])
            
            with tab1:
                df = ctx['profit_df']
                if not df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df['date'], y=df['net_profit'], name='Net Profit', marker_color=df['net_profit'].apply(lambda x: '#00C853' if x >= 0 else '#FF3D00')))
                    fig.add_trace(go.Scatter(x=df['date'], y=df['spend'], name='Ad Spend', line=dict(color='#888888', width=2, dash='dot')))
                    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                sort_mode = st.selectbox("Sort By", ["Highest Spend", "Best ROAS", "Highest CTR"], label_visibility="collapsed")
                ads = m_data.get('gallery_ads', [])
                if ads:
                    if sort_mode == "Highest Spend": ads = sorted(ads, key=lambda x: x['spend'], reverse=True)
                    elif sort_mode == "Best ROAS": ads = sorted(ads, key=lambda x: x['roas'], reverse=True)
                    elif sort_mode == "Highest CTR": ads = sorted(ads, key=lambda x: x['ctr'], reverse=True)

                    cols = st.columns(3)
                    for i, ad in enumerate(ads):
                        with cols[i % 3]:
                            img_src = ad.get('image_url') or "https://via.placeholder.com/300x300/222/888?text=No+Image"
                            roas_val = ad['roas']
                            if roas_val >= 3.0: badge_color = "rgba(0, 200, 83, 0.9); color: #fff;"
                            elif roas_val >= 1.5: badge_color = "rgba(255, 214, 0, 0.9); color: #000;"
                            else: badge_color = "rgba(255, 61, 0, 0.9); color: #fff;"
                            
                            st.markdown(f"""
                            <div class="ad-card">
                                <div class="ad-bg" style="background-image: url('{img_src}');"></div>
                                <img src="{img_src}" class="ad-image" onerror="this.src='https://via.placeholder.com/300x300/222/888?text=Video+Ad'">
                                <div class="ad-badge-top" style="background-color: {badge_color}">{roas_val}x</div>
                                <div class="ad-overlay">
                                    <div class="text-val" style="margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{ad['name']}</div>
                                    <div class="row-split">
                                        <div><div class="text-sm">Spend</div><div class="text-val">${ad['spend']:,.0f}</div></div>
                                        <div style="text-align:right;"><div class="text-sm">CTR</div><div class="text-val">{ad['ctr']:.2f}%</div></div>
                                    </div>
                                    <div class="row-split" style="margin-top:5px;">
                                        <div><div class="text-sm">CPM</div><div class="text-val">${ad['cpm']:.2f}</div></div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else: st.info("No active creatives found in this date range.")

            with tab3:
                st.dataframe(m_data['campaign_df'].sort_values("Spend", ascending=False), column_config={"Spend": st.column_config.NumberColumn(format="$%.0f"), "Sales": st.column_config.NumberColumn(format="$%.0f"), "ROAS": st.column_config.NumberColumn(format="%.2fx"), "CTR": st.column_config.NumberColumn(format="%.2f%%")}, hide_index=True, use_container_width=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)
        else: st.info("👈 Select Date Range to begin.")

# --- RIGHT: CHAT ---
with chat_col:
    with st.container(height=780, border=False):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""<div class="chat-row user-row"><div class="chat-bubble user-bubble">{msg['content']}</div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="chat-row bot-row"><div class="chat-bubble bot-bubble">{msg['content']}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br><br><br>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_memory("user", prompt)
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        context_str = ""
        if 'context' in st.session_state:
            ctx = st.session_state['context']
            s_data, m_data = ctx['shopify'], ctx['meta']
            ads_txt = "\n".join([f"{a['name']}: {a['roas']}x ROAS (${a['spend']})" for a in m_data['gallery_ads'][:10]])
            context_str = f"OVERVIEW:\nNet Profit: ${ctx['total_net_profit']:,.2f}\nRevenue: ${s_data['total_sales']}\nAd Spend: ${m_data['total_spend']}\nROAS: {ctx['roas']:.2f}x\n\nTOP ADS:\n{ads_txt}\n\nCAMPAIGNS:\n{m_data['campaign_df'].to_string(index=False)}"
        
        history = st.session_state.messages[-30:] if len(st.session_state.messages) > 30 else st.session_state.messages
        final_prompt = f"You are a Senior Media Buyer. Use this data:\n{context_str}"
        stream = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": final_prompt}] + [{"role": m["role"], "content": m["content"]} for m in history], stream=True)
        
        response_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content: response_text += chunk.choices[0].delta.content
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        save_memory("assistant", response_text)
        st.rerun()
    except Exception as e: st.error(f"Error: {e}")
