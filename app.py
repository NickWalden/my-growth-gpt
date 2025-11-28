import streamlit as st
import openai
import pandas as pd
import requests
import plotly.graph_objects as go
import json
import time
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from streamlit_gsheets import GSheetsConnection

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Growth OS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
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
        if isinstance(content, dict): content = json.dumps(content)
        new_row = pd.DataFrame([{"timestamp": datetime.now().isoformat(), "role": role, "content": content}])
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        conn.update(worksheet="ChatHistory", data=updated_data)
    except Exception: pass

# --- 3. DATA FETCHING ---
def get_product_costs(domain, token, variant_ids):
    if not variant_ids: return {}
    cost_map = {}
    try:
        unique_ids = list(set(variant_ids))
        chunks = [unique_ids[i:i + 50] for i in range(0, len(unique_ids), 50)]
        headers = {"X-Shopify-Access-Token": token}
        for chunk in chunks:
            ids_str = ",".join(map(str, chunk))
            url = f"https://{domain}/admin/api/2023-10/variants.json?ids={ids_str}&fields=id,inventory_item_id"
            res = requests.get(url, headers=headers); time.sleep(0.2)
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
        url = f"https://{domain}/admin/api/2023-10/orders.json?status=any&created_at_min={start_iso}&created_at_max={end_iso}&limit=250&fields=id,created_at,total_price,line_items,customer"
        headers = {"X-Shopify-Access-Token": token}
        all_orders = []
        while url:
            res = requests.get(url, headers=headers); time.sleep(0.3)
            if res.status_code != 200: return None, f"Shopify Error {res.status_code}"
            data = res.json()
            all_orders.extend(data.get('orders', []))
            link_header = res.headers.get('Link')
            url = None
            if link_header:
                for link in link_header.split(','):
                    if 'rel="next"' in link: url = link.split(';')[0].strip('<> ')
        
        all_vids = set()
        for o in all_orders:
            for i in o.get('line_items', []):
                if i.get('variant_id'): all_vids.add(i['variant_id'])
        real_costs = get_product_costs(domain, token, list(all_vids))
        
        daily_map = {}
        prod_sales = {}
        total_rev, total_cogs, new_cust_rev, ret_cust_rev, new_orders = 0, 0, 0, 0, 0
        
        for o in all_orders:
            date = o['created_at'][:10]
            if date not in daily_map: daily_map[date] = {'sales': 0, 'cogs': 0, 'new_sales': 0, 'ret_sales': 0}
            rev = float(o['total_price'])
            
            is_new = True
            if 'customer' in o and o['customer']:
                if int(o['customer'].get('orders_count', 1)) > 1: is_new = False
                try:
                    ord_time = datetime.fromisoformat(o['created_at'].replace('Z', '+00:00'))
                    cust_time = datetime.fromisoformat(o['customer']['created_at'].replace('Z', '+00:00'))
                    if (ord_time - cust_time).total_seconds() > 43200: is_new = False
                except: pass
            
            if is_new:
                new_cust_rev += rev; daily_map[date]['new_sales'] += rev; new_orders += 1
            else:
                ret_cust_rev += rev; daily_map[date]['ret_sales'] += rev

            cogs = 0
            for i in o.get('line_items', []):
                vid = i.get('variant_id')
                qty = i['quantity']
                price = float(i['price'])
                cost = real_costs.get(vid, price * (1 - fallback_margin))
                cogs += (cost * qty)
                prod_sales[i['title']] = prod_sales.get(i['title'], 0) + (price * qty)
            
            daily_map[date]['sales'] += rev; daily_map[date]['cogs'] += cogs
            total_rev += rev; total_cogs += cogs

        df_daily = pd.DataFrame([{'date': k, **v} for k, v in daily_map.items()])
        if not df_daily.empty:
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.sort_values('date')
            
        return {
            "daily_df": df_daily, "total_sales": total_rev, "total_cogs": total_cogs,
            "new_cust_rev": new_cust_rev, "ret_cust_rev": ret_cust_rev, "new_orders": new_orders,
            "top_products": sorted(prod_sales.items(), key=lambda x: x[1], reverse=True)[:5],
            "aov": total_rev / len(all_orders) if len(all_orders) > 0 else 0,
            "order_count": len(all_orders)
        }, None
    except Exception as e: return None, f"Shopify Crash: {e}"

def fetch_ad_creatives_batch(token, ad_ids):
    if not ad_ids: return {}
    image_map = {}
    chunks = [ad_ids[i:i + 50] for i in range(0, len(ad_ids), 50)]
    for chunk in chunks:
        try:
            ids_str = ",".join(chunk)
            url = f"https://graph.facebook.com/v17.0/?ids={ids_str}&fields=creative{{image_url,thumbnail_url,object_story_spec,asset_feed_spec,effective_object_story_id,instagram_permalink_url}}&access_token={token}"
            res = requests.get(url); time.sleep(0.2)
            if res.status_code == 200:
                data = res.json()
                for ad_id, val in data.items():
                    creative = val.get('creative', {})
                    img, link = None, None
                    try:
                        spec = creative.get('object_story_spec', {})
                        img = spec.get('link_data', {}).get('full_picture') or spec.get('link_data', {}).get('picture')
                        if not img: img = spec.get('photo_data', {}).get('url')
                        if not img: img = spec.get('video_data', {}).get('image_url')
                    except: pass
                    if not img:
                        try: img = creative.get('asset_feed_spec', {}).get('images', [])[0].get('url')
                        except: pass
                    if not img: img = creative.get('image_url') or creative.get('thumbnail_url')
                    link = creative.get('instagram_permalink_url')
                    if not link:
                        pid = creative.get('effective_object_story_id')
                        if pid: link = f"https://www.facebook.com/{pid}"
                    if img: image_map[ad_id] = {"img": img, "link": link}
        except Exception: pass
    return image_map

def fetch_meta_data(token, account_id, start_date, end_date):
    try:
        base_url = f"https://graph.facebook.com/v17.0/act_{account_id}/insights"
        time_range = json.dumps({'since': start_date.strftime('%Y-%m-%d'), 'until': end_date.strftime('%Y-%m-%d')})
        
        cmp_res = requests.get(base_url, params={'access_token': token, 'time_range': time_range, 'level': 'campaign', 'fields': 'campaign_name,spend,clicks,impressions,actions,action_values,cpm,ctr,cpc', 'limit': 100})
        daily_res = requests.get(base_url, params={'access_token': token, 'time_range': time_range, 'level': 'account', 'time_increment': 1, 'fields': 'spend,date_start', 'limit': 100})
        ad_res = requests.get(base_url, params={'access_token': token, 'time_range': time_range, 'level': 'ad', 'fields': 'ad_id,ad_name,adset_name,campaign_name,created_time,spend,ctr,cpm,actions,action_values', 'limit': 50, 'sort': ['spend_descending']})

        if cmp_res.status_code != 200: return None, f"Meta Campaign Error: {cmp_res.text}"
        if daily_res.status_code != 200: return None, f"Meta Daily Error: {daily_res.text}"
        if ad_res.status_code != 200: return None, f"Meta Ad Error: {ad_res.text}"

        cmp_data, daily_data, ad_data = cmp_res.json().get('data', []), daily_res.json().get('data', []), ad_res.json().get('data', [])
        
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
        
        gallery_ads, ad_ids_to_fetch = [], []
        for a in ad_data:
            spend = float(a.get('spend', 0))
            if spend > 0 or int(a.get('impressions', 0)) > 10:
                actions, action_values = a.get('actions', []), a.get('action_values', [])
                purchases = sum([float(x['value']) for x in actions if x['action_type'] == 'purchase'])
                revenue = sum([float(x['value']) for x in action_values if x['action_type'] == 'purchase'])
                ad_id = a.get('ad_id') or a.get('id')
                created = datetime.strptime(a.get('created_time', datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d')
                days_live = (datetime.now() - created).days
                if ad_id:
                    ad_ids_to_fetch.append(ad_id)
                    gallery_ads.append({
                        "id": ad_id, "name": a['ad_name'], "campaign": a.get('campaign_name', ''), 
                        "days_live": days_live, "spend": spend, "revenue": revenue, "purchases": int(purchases), 
                        "cpa": round(spend/purchases, 2) if purchases > 0 else 0, "roas": round(revenue/spend, 2) if spend > 0 else 0, 
                        "ctr": float(a.get('ctr', 0)), "cpm": float(a.get('cpm', 0))
                    })
        
        if ad_ids_to_fetch:
            image_map = fetch_ad_creatives_batch(token, list(set(ad_ids_to_fetch)))
            for ad in gallery_ads:
                details = image_map.get(ad['id'], {})
                ad['image_url'] = details.get('img'); ad['link'] = details.get('link')

        return {"campaign_df": pd.DataFrame(campaigns), "daily_spend_df": df_daily_spend, "total_spend": total_spend, "gallery_ads": gallery_ads}, None
    except Exception as e: return None, f"Meta Crash: {e}"

# --- 4. AI ANALYST ---
def generate_briefing(ctx, s_data, m_data):
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        analysis_payload = {
            "period": ctx.get('date_range'),
            "metrics": {"net_profit": ctx['total_net_profit'], "revenue": s_data['total_sales'], "ad_spend": m_data['total_spend'], "blended_roas": ctx['blended_mer'], "new_customers": s_data['new_orders'], "ncpa": ctx['ncpa'], "aov": s_data['aov']},
            "top_products": [p[0] for p in s_data['top_products']],
            "campaigns": m_data['campaign_df'].to_dict('records')
        }
        system_prompt = """You are an elite eCommerce Analyst. Analyze the data and return a JSON object with exactly these keys: {"headline": "A short, punchy 1-sentence summary.", "wins": ["Bullet 1", "Bullet 2"], "warnings": ["Bullet 1", "Bullet 2"], "action_plan": "One clear strategic recommendation.", "suggested_questions": ["Question 1 (Short)", "Question 2 (Short)", "Question 3 (Short)"]} Do not include markdown formatting. Return RAW JSON only."""
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(analysis_payload)}], response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)
    except Exception: return {"headline": "Analysis Unavailable", "wins": [], "warnings": [], "action_plan": "Check API keys.", "suggested_questions": []}

# --- 5. APP STATE ---
if 'messages' not in st.session_state: st.session_state.messages = load_memory()
if 'briefing' not in st.session_state: st.session_state.briefing = None
if 'logs' not in st.session_state: st.session_state.logs = []
if 'last_synced_dates' not in st.session_state: st.session_state.last_synced_dates = None

# --- 6. LAYOUT ---
header_col1, header_col2 = st.columns([5, 2], gap="medium")
with header_col1: st.markdown("# Growth OS")
with header_col2:
    preset = st.selectbox("Range Preset", ["Last 7 Days", "Last 30 Days", "This Month", "Last Month", "Custom"], index=1, label_visibility="collapsed")
    today = datetime.now().date()
    if preset == "Last 7 Days": s_d, e_d = today - timedelta(days=7), today
    elif preset == "Last 30 Days": s_d, e_d = today - timedelta(days=30), today
    elif preset == "This Month": s_d, e_d = today.replace(day=1), today
    elif preset == "Last Month": first = today.replace(day=1); e_d = first - timedelta(days=1); s_d = e_d.replace(day=1)
    else: s_d, e_d = today - timedelta(days=30), today
    date_range = st.date_input("Custom Range", value=(s_d, e_d), label_visibility="collapsed")
    if len(date_range) == 2: s_d, e_d = date_range

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    chat_width_pct = st.slider("Chat Width", 20, 60, 35, 5, format="%d%%")
    font_size = st.slider("Text Size", 12, 24, 14, 1, format="%dpx")
    margin_pct = st.slider("Margin %", 10, 90, 60, 5, format="%d%%") / 100.0
    st.divider()
    if st.button("🔄 Force Sync", type="secondary", use_container_width=True): st.session_state.last_synced_dates = None; st.rerun()
    if st.session_state.logs:
        with st.expander(f"⚠️ Logs ({len(st.session_state.logs)})"):
            for err in st.session_state.logs: st.error(err)
    st.divider()
    if st.button("Clear Memory", type="secondary", use_container_width=True): st.session_state.messages = []; st.rerun()

def run_sync_logic():
    with st.spinner("Syncing & Analyzing..."):
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
                    df_merged['mer'] = df_merged.apply(lambda row: row['sales'] / row['spend'] if row['spend'] > 0 else 0, axis=1)
                    df_merged = df_merged.sort_values('date')
                    total_net_profit = df_merged['net_profit'].sum()
                else: df_merged = pd.DataFrame(); total_net_profit = 0
                
                blended_mer = shop_data['total_sales'] / meta_data['total_spend'] if meta_data['total_spend'] > 0 else 0
                ncpa = meta_data['total_spend'] / shop_data['new_orders'] if shop_data['new_orders'] > 0 else 0
                
                ctx = {
                    "shopify": shop_data, "meta": meta_data, "profit_df": df_merged,
                    "total_net_profit": total_net_profit, "date_range": f"{s_d} to {e_d}",
                    "blended_mer": blended_mer, "ncpa": ncpa,
                    "roas": shop_data['total_sales'] / meta_data['total_spend'] if meta_data['total_spend'] > 0 else 0
                }
                st.session_state['context'] = ctx
                
                # GENERATE BRIEFING
                briefing = generate_briefing(ctx, shop_data, meta_data)
                
                # Create HTML String for Chat
                wins_html = "".join([f'<div class="briefing-item"><span class="briefing-icon">✅</span>{x}</div>' for x in briefing.get('wins', [])])
                warn_html = "".join([f'<div class="briefing-item"><span class="briefing-icon">⚠️</span>{x}</div>' for x in briefing.get('warnings', [])])
                
                briefing_html = f"""
                <div class="briefing-card">
                    <div class="briefing-head">⚡ DAILY INTELLIGENCE</div>
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #fff;">{briefing.get('headline')}</div>
                    <div style="margin-bottom: 10px;"><div style="color: #00E676; font-weight: 600; margin-bottom: 4px;">WINS</div>{wins_html}</div>
                    <div style="margin-bottom: 10px;"><div style="color: #FF3D00; font-weight: 600; margin-bottom: 4px;">WARNINGS</div>{warn_html}</div>
                    <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #333;">
                        <div style="color: #0A84FF; font-weight: 600;">RECOMMENDATION</div>
                        <div style="font-size: 13px; color: #ccc;">{briefing.get('action_plan')}</div>
                    </div>
                </div>
                """
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": briefing_html, 
                    "suggestions": briefing.get('suggested_questions', [])
                })
                save_memory("assistant", briefing_html, briefing.get('suggested_questions', []))
                
                st.session_state.last_synced_dates = (s_d, e_d)
                if not st.session_state.logs: st.toast("Sync Complete", icon="✅")
            else: st.toast("Sync Failed", icon="⚠️")
        except Exception as e: st.session_state.logs.append(f"Config Error: {e}")

if st.session_state.last_synced_dates != (s_d, e_d): run_sync_logic()

# --- 7. CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: #000; color: #fff; height: 100vh; overflow: hidden !important; }}
    header[data-testid="stHeader"] {{ background-color: transparent !important; z-index: 999; pointer-events: none; }}
    header[data-testid="stHeader"] button {{ pointer-events: auto; }}
    .stApp > header {{ background-color: transparent; }}
    .block-container {{ max-width: 100%; padding: 2rem 1rem 0 1rem; height: 100vh; overflow: hidden !important; }}
    div[data-testid="column"] {{ height: 88vh; overflow-y: auto; overflow-x: hidden; display: block; }}
    div[data-testid="column"]:nth-of-type(2) > div {{ padding-bottom: 150px !important; }}
    [data-testid="stChatInput"] {{ position: fixed !important; bottom: 0 !important; right: 1.5rem !important; left: auto !important; width: {chat_width_pct-2}% !important; min-width: 300px; background-color: #111 !important; z-index: 9999 !important; border-top: 1px solid #333; padding-top: 15px !important; padding-bottom: 25px !important; }}
    
    /* BUBBLE & TEXT SIZE */
    .chat-bubble, .chat-bubble *, .briefing-card, .briefing-card * {{ font-size: {font_size}px !important; line-height: 1.4; }}
    
    .chat-bubble {{ padding: 12px 16px; border-radius: 18px; max-width: 85%; position: relative; word-wrap: break-word; margin-bottom: 4px; display: inline-block; }}
    .user-bubble {{ background-color: #0A84FF; color: white; border-bottom-right-radius: 2px; }}
    .bot-bubble {{ background-color: #262626; color: #E5E5EA; border: 1px solid #333; border-bottom-left-radius: 2px; }}
    .chat-row {{ display: flex; margin-bottom: 12px; width: 100%; }}
    .user-row {{ justify-content: flex-end; }}
    .bot-row {{ justify-content: flex-start; }}
    div[data-testid="stMetric"] {{ background-color: #111; border: 1px solid #222; padding: 15px; border-radius: 12px; }}
    .ad-card {{ background-color: #111; border: 1px solid #222; border-radius: 12px; overflow: hidden; margin-bottom: 20px; transition: transform 0.2s; position: relative; display: flex; flex-direction: column; }}
    .ad-card:hover {{ border-color: #444; transform: translateY(-2px); z-index: 10; }}
    .ad-image-container {{ position: relative; width: 100%; height: 220px; background-color: #000; overflow: hidden; }}
    .ad-bg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-size: cover; background-position: center; filter: blur(20px) brightness(0.5); z-index: 1; }}
    .ad-image {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; z-index: 2; }}
    .ad-footer {{ background-color: #161616; padding: 12px; border-top: 1px solid #222; flex-grow: 1; }}
    .ad-title {{ font-size: 13px; font-weight: 600; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 8px; }}
    .ad-badge-top {{ position: absolute; top: 8px; right: 8px; z-index: 4; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; backdrop-filter: blur(4px); }}
    .ad-link-icon {{ position: absolute; top: 8px; left: 8px; z-index: 4; padding: 4px; border-radius: 50%; background: rgba(0,0,0,0.6); color: white; backdrop-filter: blur(4px); }}
    .grid-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 11px; color: #888; }}
    .stat-box {{ margin-bottom: 4px; }}
    .text-val {{ font-weight: 600; color: #eee; font-size: 12px; }}
    .context-tag {{ font-size: 10px; background: #222; padding: 2px 6px; border-radius: 4px; color: #888; max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 8px; }}
    .btn-view {{ display: block; width: 100%; text-align: center; background: #222; color: #ccc; font-size: 11px; padding: 6px 0; border-radius: 6px; margin-top: 8px; transition: background 0.2s; }}
    .btn-view:hover {{ background: #333; color: white; }}
    .list-row {{ display: flex; background: #111; border: 1px solid #222; border-radius: 12px; margin-bottom: 10px; overflow: hidden; transition: all 0.2s; color: inherit; text-decoration: none; }}
    .list-row:hover {{ border-color: #444; transform: translateX(4px); }}
    .list-img {{ width: 100px; height: 100px; object-fit: cover; border-right: 1px solid #222; }}
    .list-content {{ padding: 12px; flex-grow: 1; display: flex; align-items: center; justify-content: space-between; }}
    .list-info {{ flex: 1; padding-right: 20px; }}
    .list-metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: right; min-width: 250px; font-size: 11px; color: #888; }}
    .list-val {{ font-size: 13px; font-weight: 600; color: #eee; }}
    .list-badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; margin-left: 10px; }}
    
    /* BRIEFING CARD */
    .briefing-card {{ background: #1E1E1E; border: 1px solid #0A84FF; border-radius: 12px; padding: 15px; margin-bottom: 20px; }}
    .briefing-head {{ color: #0A84FF; font-weight: 700; font-size: 14px; margin-bottom: 8px; display: flex; align-items: center; }}
    .briefing-item {{ font-size: 13px; margin-bottom: 4px; display: flex; align-items: flex-start; }}
    .briefing-icon {{ margin-right: 8px; }}
    a {{ text-decoration: none; color: inherit; }}
    
    /* SUGGESTION BUTTONS */
    .stButton button {{
        border: 1px solid #333; background: #1a1a1a; color: #ddd; border-radius: 20px; font-size: 12px; padding: 4px 12px; margin-right: 5px;
    }}
    .stButton button:hover {{ border-color: #0A84FF; color: #0A84FF; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🖥️ MAIN LAYOUT
# ==========================================
dash_col, chat_col = st.columns([100-chat_width_pct, chat_width_pct], gap="medium")

# --- LEFT: DASHBOARD ---
with dash_col:
    with st.container(height=850, border=False):
        if 'context' in st.session_state:
            ctx = st.session_state['context']
            s_data, m_data = ctx['shopify'], ctx['meta']
            
            # 1. Row 1 (Revenue, Orders, AOV, True Profit)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"${s_data['total_sales']:,.0f}")
            c2.metric("Orders", f"{s_data['order_count']:,}")
            c3.metric("AOV", f"${s_data['aov']:.2f}")
            c4.metric("True Profit", f"${ctx['total_net_profit']:,.0f}", delta="Net")
            
            # 2. Row 2 (Ad Spend, MER, nCPA, ROAS)
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Ad Spend", f"${m_data['total_spend']:,.0f}")
            c6.metric("Blended MER", f"{ctx['blended_mer']:.2f}x", delta="Target: 3.0x")
            c7.metric("nCPA", f"${ctx['ncpa']:.0f}", delta="New Cust", delta_color="inverse")
            c8.metric("FB ROAS", f"{ctx['roas']:.2f}x")
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Growth (New vs Ret)", "Profit Chart", "Creative Gallery", "Campaigns"])
            with tab1:
                df = ctx['shopify']['daily_df']
                if not df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df['date'], y=df['new_sales'], name='New Customer Rev', marker_color='#007AFF'))
                    fig.add_trace(go.Bar(x=df['date'], y=df['ret_sales'], name='Returning Rev', marker_color='#BF5AF2'))
                    fig.update_layout(barmode='stack', template="plotly_dark", height=350, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig, use_container_width=True)
            with tab2:
                df = ctx['profit_df']
                if not df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df['date'], y=df['net_profit'], name='Net Profit', marker_color=df['net_profit'].apply(lambda x: '#00C853' if x >= 0 else '#FF3D00')))
                    fig.add_trace(go.Scatter(x=df['date'], y=df['spend'], name='Ad Spend', line=dict(color='#888888', width=2, dash='dot')))
                    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig, use_container_width=True)
            with tab3:
                col_view, col_sort = st.columns([1, 2])
                with col_view: view_mode = st.radio("View", ["Grid", "List"], horizontal=True, label_visibility="collapsed")
                with col_sort: sort_mode = st.selectbox("Sort", ["Highest Spend", "Best ROAS", "Most Sales"], label_visibility="collapsed")
                ads = m_data.get('gallery_ads', [])
                if ads:
                    if sort_mode == "Highest Spend": ads = sorted(ads, key=lambda x: x['spend'], reverse=True)
                    elif sort_mode == "Best ROAS": ads = sorted(ads, key=lambda x: x['roas'], reverse=True)
                    elif sort_mode == "Most Sales": ads = sorted(ads, key=lambda x: x['purchases'], reverse=True)
                    if view_mode == "Grid":
                        cols = st.columns(3)
                        for i, ad in enumerate(ads):
                            with cols[i % 3]:
                                img_src = ad.get('image_url') or "https://via.placeholder.com/300x300/222/888?text=No+Image"
                                roas_val = ad['roas']
                                badge_color = "rgba(0, 200, 83, 0.9); color: #fff;" if roas_val >= 3.0 else "rgba(255, 214, 0, 0.9); color: #000;" if roas_val >= 1.5 else "rgba(255, 61, 0, 0.9); color: #fff;"
                                link = ad.get('link') or f"https://www.facebook.com/ads/library/?id={ad['id']}"
                                st.markdown(f"""<a href="{link}" target="_blank" class="ad-link"><div class="ad-card"><div class="ad-image-container"><div class="ad-bg" style="background-image: url('{img_src}');"></div><img src="{img_src}" class="ad-image" onerror="this.src='https://via.placeholder.com/300x300/222/888?text=Video+Ad'"><div class="ad-link-icon">↗</div><div class="ad-badge-top" style="background-color: {badge_color}">{roas_val}x</div></div><div class="ad-footer"><div class="ad-title" title="{ad['name']}">{ad['name']}</div><div class="context-tag" title="Campaign: {ad['campaign']}">{ad['campaign']}</div><div class="grid-stats"><div class="stat-box">Spend <div class="text-val">${ad['spend']:,.0f}</div></div><div class="stat-box" style="text-align:right;">Rev <div class="text-val">${ad['revenue']:,.0f}</div></div><div class="stat-box">Sales <div class="text-val">{ad['purchases']}</div></div><div class="stat-box" style="text-align:right;">CPA <div class="text-val">${ad['cpa']:.2f}</div></div><div class="stat-box">CTR <div class="text-val">{ad['ctr']:.2f}%</div></div><div class="stat-box" style="text-align:right;">CPM <div class="text-val">${ad['cpm']:.2f}</div></div></div><div style="font-size:10px; color:#555; margin-top:8px; text-align:center;">Live for {ad['days_live']} days</div></div></div></a>""", unsafe_allow_html=True)
                    else:
                        for ad in ads:
                            img_src = ad.get('image_url') or "https://via.placeholder.com/100x100/222/888?text=Img"
                            roas_val = ad['roas']
                            badge_color = "#00E676" if roas_val >= 3.0 else "#FFD600" if roas_val >= 1.5 else "#FF3D00"
                            link = ad.get('link') or f"https://www.facebook.com/ads/library/?id={ad['id']}"
                            st.markdown(f"""<a href="{link}" target="_blank" class="ad-link"><div class="list-row"><img src="{img_src}" class="list-img" onerror="this.src='https://via.placeholder.com/100x100/222/888?text=Ad'"><div class="list-content"><div class="list-info"><div style="font-weight:600; color:#fff; font-size:13px; margin-bottom:4px;">{ad['name']}</div><div style="font-size:11px; color:#666;">{ad['campaign']} • Live {ad['days_live']}d</div></div><div class="list-metrics"><div>Spend <div class="list-val">${ad['spend']:,.0f}</div></div><div>Sales <div class="list-val">{ad['purchases']}</div></div><div>CPA <div class="list-val">${ad['cpa']:.2f}</div></div><div>ROAS <div class="list-val" style="color:{badge_color}">{roas_val}x</div></div></div></div></div></a>""", unsafe_allow_html=True)
                else: st.info("No active creatives found in this date range.")
            with tab4:
                st.dataframe(m_data['campaign_df'].sort_values("Spend", ascending=False), column_config={"Spend": st.column_config.NumberColumn(format="$%.0f"), "Sales": st.column_config.NumberColumn(format="$%.0f"), "ROAS": st.column_config.NumberColumn(format="%.2fx"), "CTR": st.column_config.NumberColumn(format="%.2f%%")}, hide_index=True, use_container_width=True)
            st.markdown("<br><br><br>", unsafe_allow_html=True)
        else: st.info("👈 Select Date Range to begin.")

with chat_col:
    with st.container(height=780, border=False):
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f"""<div class="chat-row user-row"><div class="chat-bubble user-bubble">{msg['content']}</div></div>""", unsafe_allow_html=True)
            else:
                if "<div" in msg['content']: 
                    st.markdown(msg['content'], unsafe_allow_html=True)
                    # Render suggestions
                    if msg.get('suggestions'):
                        cols = st.columns(len(msg['suggestions']))
                        for idx, sug in enumerate(msg['suggestions']):
                            if cols[idx].button(sug, key=f"sug_{i}_{idx}"):
                                st.session_state.trigger_prompt = sug
                                st.rerun()
                else: 
                    st.markdown(f"""<div class="chat-row bot-row"><div class="chat-bubble bot-bubble">{msg['content']}</div></div>""", unsafe_allow_html=True)
        st.markdown('<div id="end-of-chat"></div>', unsafe_allow_html=True)

# Handle Triggered Prompt
if 'trigger_prompt' in st.session_state and st.session_state.trigger_prompt:
    prompt = st.session_state.trigger_prompt
    st.session_state.trigger_prompt = None 
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_memory("user", prompt)
    run_ai = True
else:
    prompt = st.chat_input("Ask about your data...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_memory("user", prompt)
        run_ai = True
    else:
        run_ai = False

if run_ai:
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        context_str = ""
        if 'context' in st.session_state:
            ctx = st.session_state['context']
            s_data, m_data = ctx['shopify'], ctx['meta']
            ads_txt = "\n".join([f"{a['name']}: {a['roas']}x ROAS (${a['spend']})" for a in m_data['gallery_ads'][:10]])
            context_str = f"OVERVIEW:\nBlended MER: {ctx['blended_mer']:.2f}x\nNet Profit: ${ctx['total_net_profit']:,.2f}\nRevenue: ${s_data['total_sales']}\nOrders: {s_data['order_count']}\nAd Spend: ${m_data['total_spend']}\nROAS: {ctx['roas']:.2f}x\n\nNEW vs RET:\nNew Rev: ${s_data['new_cust_rev']}\nReturning: ${s_data['ret_cust_rev']}\n\nTOP ADS:\n{ads_txt}\n\nCAMPAIGNS:\n{m_data['campaign_df'].to_string(index=False)}"
        
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

# --- SCROLL SCRIPT ---
js = f"""
<script>
    function scrollBottom() {{
        const chatContainer = window.parent.document.querySelector('div[data-testid="column"]:nth-of-type(2) > div > div > div');
        if (chatContainer) {{
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }}
    }}
    setTimeout(scrollBottom, 100);
    setTimeout(scrollBottom, 500);
    setTimeout(scrollBottom, 1000);
</script>
"""
components.html(js, height=0)
