from flask import Flask, render_template, request
from db import get_news
from nlp_utils import process_text, summarize_text, categorize

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
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
            except Exception as e:
                error = f"Unable to fetch news items: {e}"
                news_items = []

            for item in news_items:
                title = item.get("title") or ""
                if query.lower() in title.lower():
                    processed = process_text(title)
                    summary = summarize_text(title)
                    category = categorize(title)
                    results.append({
                        "title": title,
                        "url": item.get("url") or "#",
                        "summary": summary,
                        "category": category,
                        "tokens": processed["tokens"]
                    })
    return render_template("index.html", results=results, query=query, error=error)

if __name__ == "__main__":
    app.run(debug=True)
