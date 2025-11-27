import streamlit as st
import openai
import pandas as pd
import requests
from datetime import datetime
from streamlit_gsheets import GSheetsConnection

# --- PAGE SETUP ---
st.set_page_config(page_title="Growth GPT (Debug Mode)", layout="wide")
st.title("🛠️ Growth GPT: Debug Mode")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- GOOGLE SHEETS MEMORY ---
def load_memory():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="ChatHistory", usecols=[0, 1, 2], ttl=0)
        if df.empty or "role" not in df.columns:
            return []
        return df.to_dict("records")
    except Exception as e:
        # In debug mode, we don't want memory errors to stop us, just print it
        st.warning(f"Memory Note: Could not load history ({e})")
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
        pass # Silent fail on save during debug

# --- DIAGNOSTIC DATA FETCHING ---
def fetch_shopify_debug(domain, token):
    st.write(f"Testing Shopify Connection to: `{domain}`...")
    try:
        url = f"https://{domain}/admin/api/2023-10/orders.json?status=any&limit=250"
        headers = {"X-Shopify-Access-Token": token}
        res = requests.get(url, headers=headers)
        
        if res.status_code == 200:
            orders = res.json().get('orders', [])
            sales = sum([float(o['total_price']) for o in orders])
            st.success(f"✅ Shopify Success! Found {len(orders)} orders.")
            return {"sales": sales, "orders": len(orders), "aov": sales/len(orders) if orders else 0}
        else:
            st.error(f"❌ Shopify Error {res.status_code}: {res.text}")
            return None
    except Exception as e:
        st.error(f"❌ Shopify Critical Fail: {e}")
        return None

def fetch_meta_debug(token, account_id):
    st.write(f"Testing Meta Connection to Account ID: `{account_id}`...")
    try:
        # Note: Ensure account_id is just numbers
        url = f"https://graph.facebook.com/v17.0/act_{account_id}/insights"
        params = {'access_token': token, 'date_preset': 'last_30d', 'fields': 'spend,clicks,impressions'}
        res = requests.get(url, params=params)
        
        if res.status_code == 200:
            data = res.json().get('data', [])
            if data:
                summary = data[0]
                st.success(f"✅ Meta Success! Spend: ${summary.get('spend')}")
                return {
                    "spend": float(summary.get('spend', 0)),
                    "clicks": int(summary.get('clicks', 0)),
                    "impressions": int(summary.get('impressions', 0))
                }
            else:
                st.warning("⚠️ Meta connected, but returned NO data for last 30d. (Is the account active?)")
                return {"spend": 0, "clicks": 0, "impressions": 0}
        else:
            st.error(f"❌ Meta Error {res.status_code}: {res.text}")
            return None
    except Exception as e:
        st.error(f"❌ Meta Critical Fail: {e}")
        return None

# --- MAIN LOGIC ---
if 'messages' not in st.session_state:
    st.session_state.messages = load_memory()

# Fetch Data Button
if st.button("🔄 DEBUG: Analyze Live Data"):
    # Load keys from Secrets
    try:
        s_domain = st.secrets["SHOPIFY_DOMAIN"]
        s_token = st.secrets["SHOPIFY_TOKEN"]
        m_token = st.secrets["META_TOKEN"]
        m_id = st.secrets["META_ACCOUNT_ID"]
        
        st.info("Keys loaded from Secrets. Testing connections now...")
        
        shop_data = fetch_shopify_debug(s_domain, s_token)
        meta_data = fetch_meta_debug(m_token, m_id)
        
        if shop_data and meta_data:
            roas = shop_data['sales'] / meta_data['spend'] if meta_data['spend'] > 0 else 0
            st.session_state['context'] = {
                "shopify": shop_data, 
                "meta": meta_data, 
                "roas": round(roas, 2)
            }
            st.success("✅ INTEGRATION COMPLETE: Data Loaded.")
        
    except FileNotFoundError:
        st.error("Secrets file not found. Please check Streamlit Cloud Settings.")
    except KeyError as e:
        st.error(f"Missing Key in Secrets: {e}")

# Display Dashboard
if 'context' in st.session_state:
    ctx = st.session_state['context']
    c1, c2, c3 = st.columns(3)
    c1.metric("Sales (30d)", f"${ctx['shopify']['sales']:,.2f}")
    c2.metric("Ad Spend", f"${ctx['meta']['spend']:,.2f}")
    c3.metric("ROAS", f"{ctx['roas']}x")

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your ads..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    save_memory("user", prompt)
    
    # AI Logic
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        context_str = ""
        if 'context' in st.session_state:
            context_str = f"CURRENT DATA: {st.session_state['context']}"
            
        system_prompt = f"You are an expert Media Buyer. {context_str}. Answer briefly."
        
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}] + 
                     [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]],
            stream=True
        )
        
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_memory("assistant", response)
        
    except Exception as e:
        st.error(f"AI Error: {e}")
