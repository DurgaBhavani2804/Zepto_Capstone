import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3 

books = []

for page in range(1, 6):  # scrape 5 pages → ≥ 60 books
    url = f"https://books.toscrape.com/catalogue/page-{page}.html"
    response = requests.get(url)
    response.encoding = "utf-8"   # ✅ force UTF-8 decoding
    soup = BeautifulSoup(response.text, "html.parser")

    for book in soup.select(".product_pod"):
        title = book.h3.a["title"]
        price = book.select_one(".price_color").text
        star_rating = book.p["class"][1]   # e.g. "Three"
        availability = book.select_one(".availability").text.strip()
        category = "All products"

        books.append([title, price, star_rating, availability, category])




df = pd.DataFrame(books, columns=["title","price","star_rating","availability","category"])


df["price_gbp"] = df["price"].str.replace("£","").astype(float)
rating_map = {"One":1,"Two":2,"Three":3,"Four":4,"Five":5}
df["rating"] = df["star_rating"].map(rating_map)
df["in_stock"] = df["availability"].str.contains("In stock")


df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
df["rating"] = df["rating"].fillna(df["rating"].median())
df = df.dropna(subset=["title","availability","category"])


conversion_rate = 105.50
df["price_inr"] = df["price_gbp"] * conversion_rate

print(df.head())
print(f"Final dataset size: {len(df)}")        




conn = sqlite3.connect("books.db")
cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
)""")


categories = df["category"].unique()
for cat in categories:
    cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,))


for _, row in df.iterrows():
    cursor.execute("SELECT category_id FROM categories WHERE category_name=?", (row["category"],))
    cat_id = cursor.fetchone()[0]
    cursor.execute("""INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
                      VALUES (?, ?, ?, ?, ?, ?)""",
                   (row["title"], row["price_gbp"], row["price_inr"], int(row["rating"]), int(row["in_stock"]), cat_id))

conn.commit()


print("\nTop 5 expensive books:")
print(pd.read_sql("SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 5", conn))

print("\nAverage price per rating:")
print(pd.read_sql("SELECT rating, AVG(price_gbp) AS avg_price FROM books GROUP BY rating", conn))

print("\nDistinct categories:")
print(pd.read_sql("SELECT DISTINCT category_name FROM categories", conn))

print("\nBooks priced between 40 and 50 GBP:")
print(pd.read_sql("SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 40 AND 50", conn))

print("\nJOIN query (highest-rated books per category):")
join_query = """
SELECT b.title, b.rating, c.category_name
FROM books b
JOIN categories c ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.price_gbp DESC
LIMIT 10;
"""
join_df = pd.read_sql(join_query, conn)
print(join_df)



books_df = pd.read_sql("SELECT * FROM books", conn)
cats_df = pd.read_sql("SELECT * FROM categories", conn)

merged_df = pd.merge(books_df, cats_df, left_on="category_id", right_on="category_id")

print("\nPandas merge output (JOIN replication):")
print(
    merged_df[["title", "rating", "category_name", "price_gbp"]]
    .sort_values(["rating", "price_gbp"], ascending=[False, False])
    .head(10)
)

# Pipeline updated for capstone submission
