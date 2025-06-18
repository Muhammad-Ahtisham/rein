import streamlit as st
import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import requests
from PIL import Image
from io import BytesIO
import random
import torch
import torch.nn as nn
import torch.optim as optim
# ------------------ SETUP ------------------
st.set_page_config(page_title="Product Recommendation", layout="centered")
st.title("🔍 Surgical Tool Recommendation System")

# ---------- DATABASE CONNECTION ----------
conn = sqlite3.connect("recommendation.db", check_same_thread=False)
cursor = conn.cursor()

# ---------- INIT DB TABLES ----------
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    userID TEXT PRIMARY KEY,
    previousPurchases TEXT,
    category TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS tools (
    Title TEXT,
    Title_URL TEXT,
    Image TEXT,
    Material TEXT,
    Length TEXT,
    Use TEXT,
    Brand TEXT,
    Category TEXT
)''')

conn.commit()

# ---------- LOAD DATA FROM DATABASE ----------
@st.cache_data(show_spinner=False)
def load_data_fresh():
    user_df = pd.read_sql_query("SELECT * FROM users", conn)
    tools_df = pd.read_sql_query("SELECT * FROM tools", conn)
    return user_df, tools_df

# ---------- FIND MATCH FUNCTION ----------
def find_best_match(prod_name, choices, threshold=70):
    match, score = process.extractOne(prod_name.lower().strip(), choices)
    if score >= threshold:
        return match
    return None

# ---------- IMAGE DISPLAY ----------
def display_resized_image(image_url, max_width=300):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        w_percent = max_width / float(img.size[0])
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((max_width, h_size), Image.LANCZOS)
        st.image(img, use_container_width=False)
    except:
        st.write("🖼️ Image unavailable")

# ---------- DQN MODEL ----------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ---------- RL FUNCTIONS (DQN) ----------
dqn_model = DQN(input_dim=1, output_dim=1)
optimizer = optim.Adam(dqn_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

memory = []  # (state, action, reward)

def dqn_recommendation(user_id, category, tools_df):
    available_tools = tools_df[tools_df['Category'].str.lower().str.contains(category.lower())]['Title'].unique().tolist()
    if not available_tools:
        return None
    state = torch.tensor([hash(user_id + category) % 1000], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = [dqn_model(state).item() for _ in available_tools]
    max_index = q_values.index(max(q_values))
    return available_tools[max_index]

def update_dqn(user_id, tool, category, reward):
    state = torch.tensor([hash(user_id + category) % 1000], dtype=torch.float32).unsqueeze(0)
    target = torch.tensor([[reward]], dtype=torch.float32)
    output = dqn_model(state)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------- DATA PIPELINE FUNCTION ----------
def get_updated_data():
    df, tools_df = load_data_fresh()
    purchase_matrix = df.set_index('userID')['previousPurchases'].str.get_dummies(sep='|')
    sim_matrix = cosine_similarity(purchase_matrix.values)
    sim_df = pd.DataFrame(sim_matrix, index=purchase_matrix.index, columns=purchase_matrix.index)
    tools_df['Title_clean'] = tools_df['Title'].str.lower().str.strip()
    product_choices = tools_df['Title_clean'].tolist()
    return df, tools_df, purchase_matrix, sim_df, product_choices

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["📊 Recommend Products", "➕ Add New User", "🧠 Content-Based Suggestions"])

# ========== TAB 1: USER-BASED RL RECOMMENDATION ==========
with tab1:
    df, tools_df, purchase_matrix, sim_df, product_choices = get_updated_data()
    st.write("## 📌 RL-Based Product Recommendations")
    user_list = list(purchase_matrix.index)
    selected_user = st.selectbox("Select a User ID", user_list)
    custom_user_input = st.text_input("Or enter a User ID manually:", value=selected_user)

    if custom_user_input in purchase_matrix.index:
        selected_user = custom_user_input
        category = df[df['userID'] == selected_user]['category'].values[0]
        recommended_tool = dqn_recommendation(selected_user, category, tools_df)

        if recommended_tool:
            row = tools_df[tools_df['Title'] == recommended_tool].iloc[0]
            st.markdown(f"### [{row['Title']}]({row['Title_URL']})")
            display_resized_image(row['Image'])

            if st.button("👍 Mark as Useful", key=f"{selected_user}_{recommended_tool}"):
                update_dqn(selected_user, recommended_tool, category, reward=1)
                st.success("Feedback updated with reward = 1")
        else:
            st.warning("No recommendation available.")
    else:
        st.warning("User ID not found in the dataset.")

# ========== TAB 2: ADD NEW USER ==========
with tab2:
    st.write("## ➕ Create a New User Profile")
    new_user_id = st.text_input("🔹 Enter New User ID")
    new_user_purchases = st.text_input("🔹 Purchased tools (use '|' to separate multiple items):")
    new_user_category = st.text_input("🔹 Enter Tool Category (e.g., Cutting, Grasping)")

    if st.button("✅ Add User and Generate Recommendations"):
        if new_user_id.strip() == "" or new_user_purchases.strip() == "" or new_user_category.strip() == "":
            st.warning("Please enter User ID, purchases, and category.")
        else:
            cursor.execute("SELECT COUNT(*) FROM users WHERE userID=?", (new_user_id,))
            if cursor.fetchone()[0] > 0:
                st.warning("User ID already exists. Please choose another one.")
            else:
                cursor.execute("INSERT INTO users (userID, previousPurchases, category) VALUES (?, ?, ?)",
                               (new_user_id.strip(), new_user_purchases.strip(), new_user_category.strip()))
                conn.commit()
                st.success(f"User '{new_user_id}' added successfully!")
                st.cache_data.clear()

# ========== TAB 3: CONTENT-BASED FILTERING ==========
with tab3:
    st.write("## 🧠 Content-Based Filtering")

    tools_df = load_data_fresh()[1]
    tools_df['Title_clean'] = tools_df['Title'].str.lower().str.strip()
    selected_tool = st.selectbox("🔍 Select a Tool to Find Similar Ones", tools_df['Title'])
    selected_category = st.text_input("🔍 Enter Your Category for Personalized Recommendations")

    selected_row = tools_df[tools_df['Title'] == selected_tool].iloc[0]
    st.markdown(f"### 🔧 You selected: {selected_row['Title']}")
    display_resized_image(selected_row['Image'])

    st.subheader("🔗 Similar Products (Same Category):")
    selected_category_from_tool = selected_row.get("Category", "").lower().strip()

    if selected_category_from_tool:
        similar_products = tools_df[tools_df['Category'].str.lower().str.strip() == selected_category_from_tool]
        similar_products = similar_products[similar_products['Title'] != selected_row['Title']].head(5)

        if similar_products.empty:
            st.info("No similar products found in the same category.")
        else:
            for _, row in similar_products.iterrows():
                st.markdown(f"### [{row['Title']}]({row['Title_URL']})")
                display_resized_image(row['Image'])
    else:
        st.warning("Category information is missing for the selected tool.")

    if selected_category:
        st.subheader(f"🩺 Products for Category: {selected_category}")
        cat_prods = tools_df[tools_df['Category'].str.lower().str.contains(selected_category.lower())]
        if cat_prods.empty:
            st.info("No products found for this category.")
        else:
            for _, row in cat_prods.head(5).iterrows():
                st.markdown(f"### [{row['Title']}]({row['Title_URL']})")
                display_resized_image(row['Image'])
