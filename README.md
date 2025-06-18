# Purchase History-Based Recommender App

This is a Streamlit web application for recommending products to users based on purchase history similarity using cosine similarity.

## Features

- Upload Excel file with columns: `userID` and `previousPurchases`
- Choose a user or enter one manually to get top product recommendations
- Uses one-hot encoding and cosine similarity

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sample Data Format

| userID | previousPurchases           |
|--------|-----------------------------|
| U1     | scalpel|gloves              |
| U2     | sutures|scalpel             |
| U3     | gloves|sutures              |