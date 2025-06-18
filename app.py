import streamlit as st
import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import requests
from PIL import Image
from io import BytesIO

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

cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (
    userID TEXT,
    toolTitle TEXT,
    reward INTEGER,
    PRIMARY KEY (userID, toolTitle)
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Recommend Products", 
    "➕ Add New User", 
    "🧠 Content-Based Suggestions", 
    "📋 Feedback History"
])

# ========== TAB 1: USER-BASED RECOMMENDATION ==========
with tab1:
    df, tools_df, purchase_matrix, sim_df, product_choices = get_updated_data()
    st.write("## 📌 User-Based Product Recommendations")
    user_list = list(purchase_matrix.index)
    selected_user = st.selectbox("Select a User ID", user_list)
    custom_user_input = st.text_input("Or enter a User ID manually:", value=selected_user)

    if custom_user_input in purchase_matrix.index:
        selected_user = custom_user_input
        sim_scores = sim_df[selected_user].drop(selected_user)
        sim_scores = sim_scores[sim_scores > 0]

        if sim_scores.empty:
            st.write("No similar users found for this user.")
        else:
            weighted_scores = purchase_matrix.loc[sim_scores.index].T.dot(sim_scores)
            user_vector = purchase_matrix.loc[selected_user]
            new_scores = weighted_scores[user_vector == 0]

            # 🧠 Reinforcement: Adjust recommendation scores based on user feedback
            recommendation_scores = []
            for prod in new_scores.index:
                cursor.execute("SELECT reward FROM feedback WHERE userID=? AND toolTitle=?", (selected_user, prod))
                result = cursor.fetchone()
                reward = result[0] if result else 0
                adjusted_score = new_scores[prod] + reward
                recommendation_scores.append((prod, adjusted_score))

            top5 = sorted(recommendation_scores, key=lambda x: x[1], reverse=True)[:5]

            if not top5:
                st.write("No new product recommendations available for this user.")
            else:
                st.subheader("🎯 Top 5 Recommended Products:")
                for prod, _ in top5:
                    best_match = find_best_match(prod, product_choices)
                    if best_match:
                        row = tools_df[tools_df['Title_clean'] == best_match].iloc[0]
                        st.markdown(f"### [{prod}]({row['Title_URL']})")
                        display_resized_image(row['Image'])

                        feedback_key_pos = f"{selected_user}_{prod}_pos"
                        feedback_key_neg = f"{selected_user}_{prod}_neg"

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍 Mark as Useful", key=feedback_key_pos):
                                cursor.execute("INSERT OR REPLACE INTO feedback (userID, toolTitle, reward) VALUES (?, ?, ?)",
                                               (selected_user, prod, 1))
                                conn.commit()
                                st.success(f"✅ Positive feedback recorded for '{prod}'!")
                        with col2:
                            if st.button("👎 Mark as Not Useful", key=feedback_key_neg):
                                cursor.execute("INSERT OR REPLACE INTO feedback (userID, toolTitle, reward) VALUES (?, ?, ?)",
                                               (selected_user, prod, -1))
                                conn.commit()
                                st.warning(f"❌ Negative feedback recorded for '{prod}'!")
                    else:
                        st.write(f"- {prod} (No match found)")
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

# ========== TAB 4: FEEDBACK HISTORY ==========
with tab4:
    st.write("## 📋 User Feedback History")
    feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
    if feedback_df.empty:
        st.info("No feedback data available yet.")
    else:
        st.dataframe(feedback_df)

        st.write("### 📊 Feedback Summary by Tool")
        summary = feedback_df.groupby('toolTitle')['reward'].sum().reset_index()
        summary['Feedback Type'] = summary['reward'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

        feedback_chart = summary.groupby(['toolTitle', 'Feedback Type']).size().unstack(fill_value=0)
        st.bar_chart(feedback_chart)
