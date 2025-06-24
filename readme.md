# E-commerce Product Recommendation System

A Streamlit app for product recommendations using text similarity, price, and ratings.

---

## 🚀 How to Run the App

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place `app.py` and `data.json` in the same directory.

**Sample `data.json` format:**

```json
[
  {
    "product_name": "Example",
    "price": 99.99,
    "category": "Electronics",
    "description": "Description",
    "rating": 4.5
  }
]
```

### 3. Run the App

```bash
streamlit run app.py
```

Access the app at [http://localhost:8501](http://localhost:8501).

---

## 💡 Use

- Filter by category or budget  
- Search products  
- Navigate pages  
- Select a product to view recommendations

---

## 🤖 AI Feature Chosen

**Recommendation System**  
Uses TF-IDF and cosine similarity to recommend products based on:

- Product name  
- Description  
- Category  

Weighted with:

- **Rating** (30%)  
- **Price** (20%)

📌 *Example: Similar to Amazon’s related product suggestions.*

---

## 🛠️ Tools & Libraries Used

- **Streamlit**: Web interface  
- **Pandas**: Data handling  
- **Scikit-learn**: TF-IDF and cosine similarity  
- **JSON**: Data loading

---

## 📌 Notable Assumptions

- `data.json` exists with valid product data  
- Prices are non-negative  
- Ratings range from 0 to 5  
- Categories are consistent in `data.json`  
