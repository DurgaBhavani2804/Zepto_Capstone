This project scrapes book data from Books to Scrape website.
It cleans the data, converts prices, and saves everything into a SQLite database.
Finally, it runs some SQL queries and shows the results.

## Requirements
 pip install -r requirements.txt

## clone the repo:
git clone https://github.com/DurgaBhavani2804/Zepto_Capstone.git
cd zepto_project/data_pipeline

## Create virtual environment:
python -m venv venv

## Activate it:
venv\Scripts\activate
source venv/bin/activate

## Packages I used :
pip install -r requirements.txt

## How I run:
python scraper.py

## After Run my code It will gave me :

Scrape 100 books (≥ 60 required).
Clean the data.
Convert GBP → INR.
Create books.db file.
Run SQL queries and print results.
Show JOIN outputs using both pd.read_sql and pd.merge.

## fixed Conversion :
1 GBP = 105.50 INR


## Cleaning Decisions:

Removed £ symbol from price.
Converted star ratings (“One”, “Two”, …) → numbers (1–5).
Availability → boolean (True if “In stock”).
Filled missing numbers with median.
Dropped rows with missing text fields.

## Database Schema :

categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL,
rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))

## Git Workflow:

git checkout -b feature/data-pipeline
git add scraper.py books.db README.md
git commit -m "Initial pipeline commit"
git commit -m "Added SQL queries + outputs"
git checkout main
git merge feature/data-pipeline

