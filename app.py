"""
    Recommendation System
    Based on user selections or profile (e.g. preferred categories, budget), suggest 3-5 relevant products.
    Rule-based
"""

#import necessary libraries
import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="E-commerce Product Recommendation", layout="wide")

# Read product data from JSON
try:
    with open('data.json', 'r') as file:
        products_data = json.load(file)
except FileNotFoundError:
    st.error("Error: data.json file not found.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error: Invalid JSON format in data.json.")
    st.stop()

# Extract unique categories
categories = sorted(list(set(product['category'] for product in products_data)))

# Prepare combined text for TF-IDF
for product in products_data:
    product['combined_text'] = f"{product['product_name']} {product['description']} {product['category']}"

# Calculate combined score for recommendations
def calculate_combined_score(product, max_budget, tfidf_score):
    budget_mid = max_budget / 2
    max_price = max(p['price'] for p in products_data)
    price_diff = abs(product['price'] - budget_mid)
    price_score = 1 - (price_diff / max_price) if max_price > 0 else 1
    rating_score = product['rating'] / 5
    return 0.5 * tfidf_score + 0.3 * rating_score + 0.2 * price_score

# Get recommendations based on selected product
def get_recommendations(selected_product, category, max_budget):
    filtered_products = [
        product for product in products_data
        if product['category'] == category and 
           product['price'] <= max_budget and 
           product['product_name'] != selected_product['product_name']
    ]
    
    if not filtered_products:
        return []
    
    texts = [product['combined_text'] for product in filtered_products]
    texts.append(selected_product['combined_text'])
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    for i, product in enumerate(filtered_products):
        product['tfidf_score'] = cosine_sim[i]
        product['combined_score'] = calculate_combined_score(product, max_budget, cosine_sim[i])
    
    filtered_products.sort(key=lambda x: x['combined_score'], reverse=True)
    return filtered_products[:4]

# Streamlit UI
st.title("E-commerce Product Recommendation System")

# Sidebar for filters
st.sidebar.header("Filters")
selected_category = st.sidebar.selectbox("Select Category", ["All"] + categories, key="category")
max_budget = st.sidebar.slider("Maximum Budget ($)", 0.0, 500.0, 500.0, step=0.01, key="budget")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'filter_hash' not in st.session_state:
    st.session_state.filter_hash = ""

# Search bar
search_query = st.text_input("Search for a product (e.g., 'wireless headphones')", key="search")

# Create a hash of current filters to detect changes
current_filter_hash = f"{selected_category}_{max_budget}_{search_query}"

# Reset page to 1 if filters change
if current_filter_hash != st.session_state.filter_hash:
    st.session_state.page = 1
    st.session_state.filter_hash = current_filter_hash

# Filter products based on search and filters
filtered_products = products_data
if selected_category != "All":
    filtered_products = [p for p in filtered_products if p['category'] == selected_category]
filtered_products = [p for p in filtered_products if p['price'] <= max_budget]
if search_query.strip():
    search_query = search_query.lower()
    filtered_products = [
        p for p in filtered_products
        if search_query in p['product_name'].lower() or search_query in p['description'].lower()
    ]

# Pagination settings
items_per_page = 10
total_items = len(filtered_products)
total_pages = (total_items + items_per_page - 1) // items_per_page
current_page = st.number_input("Page", min_value=1, max_value=max(total_pages, 1), value=st.session_state.page, step=1, key="page_input")

# Update session state page when number input changes
if current_page != st.session_state.page:
    st.session_state.page = current_page
    st.rerun()

# Calculate start and end indices
start_idx = (current_page - 1) * items_per_page
end_idx = min(start_idx + items_per_page, total_items)

# Display search results
st.header(f"Search Results (Page {current_page} of {total_pages})")
if not filtered_products:
    st.warning("No products found matching your criteria.")
else:
    paginated_products = filtered_products[start_idx:end_idx]
    
    df = pd.DataFrame(paginated_products)[['product_name', 'price', 'category', 'description', 'rating']]
    df.columns = ['Product Name', 'Price ($)', 'Category', 'Description', 'Rating']
    df['Price ($)'] = df['Price ($)'].round(2)
    df['Rating'] = df['Rating'].map(lambda x: f"{x}/5")
    
    selected_index = st.dataframe(
        df,
        use_container_width=True,
        height=300,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if current_page > 1:
            if st.button("Previous"):
                st.session_state.page = current_page - 1
                st.rerun()
    with col3:
        if current_page < total_pages:
            if st.button("Next"):
                st.session_state.page = current_page + 1
                st.rerun()
    
    if selected_index.get("selection", {}).get("rows"):
        selected_row = selected_index["selection"]["rows"][0]
        selected_product = paginated_products[selected_row]
        
        st.subheader("Selected Product")
        st.write(f"**Name**: {selected_product['product_name']}")
        st.write(f"**Price**: ${selected_product['price']:.2f}")
        st.write(f"**Category**: {selected_product['category']}")
        st.write(f"**Description**: {selected_product['description']}")
        st.write(f"**Rating**: {selected_product['rating']}/5")
        
        st.subheader("Recommended Products")
        recommendations = get_recommendations(
            selected_product, 
            selected_category if selected_category != "All" else selected_product['category'], 
            max_budget
        )
        
        if not recommendations:
            st.warning("No recommendations found matching your criteria.")
        else:
            rec_df = pd.DataFrame(recommendations)[['product_name', 'price', 'category', 'description', 'rating']]
            rec_df.columns = ['Product Name', 'Price ($)', 'Category', 'Description', 'Rating']
            rec_df['Price ($)'] = rec_df['Price ($)'].round(2)
            rec_df['Rating'] = rec_df['Rating'].map(lambda x: f"{x}/5")
            st.dataframe(rec_df, use_container_width=True, height=200)