import os
from flask import Flask, render_template, request
from db import get_news
from nlp_utils import process_text, summarize_text, categorize

app = Flask("NLP News App")

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/health")
def health():
    return {"status": "ok"}, 200

@app.route("/news", methods=["GET", "POST"])
def news_search():
    results = []
    query = ""
    error = None
    if request.method == "POST":
        query = (request.form.get("query") or "").strip()
        if not query:
            error = "Please enter a search term."
        else:
            try:
                news_items = get_news()
                if not news_items:
                    error = "No news items available. Please try again later."
                    news_items = []
            except Exception as e:
                error = f"Database connection error. Please try again later."
                news_items = []
                print(f"Database error: {e}")

            if news_items:
                try:
                    seen_titles = set()  # Track seen titles to avoid duplicates
                    for item in news_items:
                        try:
                            title = item.get("title") or ""
                            category = categorize(title)
                            if ((query.lower() in title.lower()) or (query.lower() in category.lower())) and title not in seen_titles:
                                seen_titles.add(title)
                                processed = process_text(title)
                                summary = summarize_text(title)
                                results.append({
                                    "title": title,
                                    "url": item.get("url") or "#",
                                    "summary": summary,
                                    "category": category,
                                    "tokens": processed["tokens"]
                                })
                        except Exception as item_error:
                            print(f"Error processing item: {item_error}")
                            continue
                except Exception as e:
                    error = f"Error processing results: {str(e)}"
                    print(f"Processing error: {e}")
    return render_template("index.html", results=results, query=query, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
