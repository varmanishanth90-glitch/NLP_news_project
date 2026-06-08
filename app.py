import os
from flask import Flask, render_template, request
from db import get_news
from nlp_utils import process_text, summarize_text, categorize
from next_word_predictor import predict_next_words, get_sample_prompts

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
    sample_data = []
    
    # Always fetch sample data for display
    try:
        all_news = get_news()
        if all_news:
            sample_data = []
            for item in all_news[:30]:  # Get first 30 records
                try:
                    title = item.get("title") or ""
                    description = item.get("description") or ""
                    content = item.get("content") or ""
                    category = categorize(title, description, content)
                    sample_data.append({
                        "title": title,
                        "url": item.get("url") or "#",
                        "category": category,
                        "source": item.get("source_name") or "Unknown",
                        "description": description[:100] if description else ""
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"Error fetching sample data: {e}")
    
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
                            description = item.get("description") or ""
                            content = item.get("content") or ""
                            category = categorize(title, description, content)
                            
                            # Search across all text fields
                            search_text = f"{title} {description} {content} {category}".lower()
                            if (query.lower() in search_text) and title not in seen_titles:
                                seen_titles.add(title)
                                # Combine all text for better summarization
                                combined_text = f"{title} {description} {content}"
                                processed = process_text(combined_text)
                                summary = summarize_text(combined_text)
                                results.append({
                                    "title": title,
                                    "url": item.get("url") or "#",
                                    "summary": summary,
                                    "category": category,
                                    "description": description,
                                    "content": content[:200] if content else "",
                                    "tokens": processed["tokens"]
                                })
                        except Exception as item_error:
                            print(f"Error processing item: {item_error}")
                            continue
                except Exception as e:
                    error = f"Error processing results: {str(e)}"
                    print(f"Processing error: {e}")
    return render_template("index.html", results=results, query=query, error=error, sample_data=sample_data)

@app.route("/nextword", methods=["GET", "POST"])
def next_word_predictor():
    prompt = ""
    suggestions = []
    if request.method == "POST":
        prompt = (request.form.get("prompt") or "").strip()
        if prompt:
            suggestions = predict_next_words(prompt, top_n=5)
    examples = get_sample_prompts(5)
    return render_template("nextword.html", prompt=prompt, suggestions=suggestions, examples=examples)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
