# NLP News Project

A Flask-based web application that categorizes news articles using Natural Language Processing (NLP) and provides next-word prediction functionality.

## Features

- **News Categorization**: Automatically classify news articles into 10 categories:
  - Finance, Sports, Politics, Technology, Health
  - Entertainment, War, Automobile, Airlines, Education

- **Next-Word Prediction**: Suggests the next words based on user input

- **Text Processing**: Tokenization, stemming, lemmatization of news content

- **News Search**: Search and filter news articles by category and keywords

- **Responsive UI**: Clean, modern web interface with voice input support

## Project Structure

```
news-nlp-project/
├── app.py                    # Main Flask application
├── nlp_utils.py              # NLP utilities and models
├── next_word_predictor.py    # Next-word prediction model
├── db.py                     # Database connections
├── train_models.py           # Model training script
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version for deployment
├── Procfile                  # Deployment configuration
├── data/
│   └── next_word_dataset.txt # Training data for word prediction
├── nltk_data/                # NLTK language resources
├── templates/                # HTML templates
│   ├── landing.html
│   ├── index.html
│   └── nextword.html
└── static/                   # Static files
    ├── style.css
    └── voice.js
```

## Installation

### Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd news-nlp-project
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file:
```env
MYSQL_HOST=your_host
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database
```

5. **Train models** (optional, auto-trains on first run)
```bash
python train_models.py
```

6. **Run the app**
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

## Deployment to Render

### Prerequisites
- GitHub account with your code pushed
- MySQL database (can use Render's PostgreSQL or external MySQL)
- Render account (https://render.com)

### Steps

1. **Push code to GitHub**
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Create Web Service on Render**
- Go to https://render.com/dashboard
- Click "New" → "Web Service"
- Connect GitHub repository
- Select repository and branch (main/master)

3. **Configure on Render**

| Setting | Value |
|---------|-------|
| Name | news-nlp-app |
| Environment | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |
| Plan | Free or Pro |

4. **Set Environment Variables**
In Render dashboard → Environment:
```
MYSQL_HOST=your_mysql_host
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=your_mysql_database
```

5. **Deploy**
- Click "Create Web Service"
- Render deploys automatically
- Your app will be live at `https://your-app-name.onrender.com`

## Usage

### Via Web Interface

1. **News Search**: 
   - Enter search term
   - View categorized results
   - Click on articles to view details

2. **Next-Word Prediction**:
   - Type a word or phrase
   - Get suggestions for next words
   - Use voice input (click microphone icon)

### Via API

**Categorize news:**
```python
from nlp_utils import categorize

category = categorize(
    title="Apple releases new iPhone",
    description="Latest model features",
    content="Full article content..."
)
# Returns: "Technology"
```

**Predict next words:**
```python
from next_word_predictor import predict_next_words

suggestions = predict_next_words("The stock market", top_n=5)
# Returns: ["surged", "rose", "fell", "continued", "reached"]
```

**Process text:**
```python
from nlp_utils import process_text

result = process_text("Your text here")
# Returns: {"tokens": [...], "stems": [...], "lemmas": [...]}
```

## Model Training

Train or retrain models with:
```bash
python train_models.py
```

This will:
- Train category classification model (Naive Bayes)
- Train next-word prediction model (Bigram)
- Test accuracy on sample data
- Display model statistics

### Current Model Performance
- **Accuracy**: 88.9% on test samples
- **Vocabulary**: 489 unique words
- **Categories**: 12 news categories
- **Model Size**: ~50 KB

## Adding More Training Data

### For Category Classification
Edit `nlp_utils.py` and add to `CATEGORY_TRAINING_DATA`:
```python
CATEGORY_TRAINING_DATA = [
    ("Your news headline", "Category"),
    # Add more examples...
]
```

### For Next-Word Prediction
Add lines to `data/next_word_dataset.txt`:
```
word sequence one
word sequence two
More training phrases here
```

Then retrain:
```bash
python train_models.py
```

## Technologies Used

- **Flask**: Web framework
- **NLTK**: Natural Language Toolkit
- **MySQL**: Database
- **Gunicorn**: Production server
- **Python 3.10+**: Programming language

## Dependencies

See `requirements.txt`:
- Flask
- mysql-connector-python
- python-dotenv
- nltk
- speechrecognition
- gunicorn

## Model Files

- **category_model.pkl**: Naive Bayes classifier (~30 KB)
  - Contains word counts, priors, vocabulary
  - Auto-generated on first run or by `train_models.py`

- **bigram_model.pkl**: N-gram model for word prediction (~30 KB)
  - Contains bigram frequencies
  - Auto-generated on first run or by `train_models.py`

## Troubleshooting

### Models not training
- Ensure NLTK data is downloaded
- Check `nltk_data/` directory exists
- Run: `python train_models.py -v` for verbose output

### Database connection error
- Verify MySQL credentials in `.env`
- Check database is accessible
- Ensure all required tables exist

### App not starting
- Install all dependencies: `pip install -r requirements.txt`
- Check Python 3.10+ is installed
- View error logs for specific issues

### Slow first load on Render
- Models auto-training on startup (normal)
- Takes ~30 seconds first time
- Subsequent loads are fast (~100ms)

## Performance

- Model training: ~5-10 seconds
- Article categorization: ~20ms per article
- Next-word prediction: ~10ms per query
- Database query: ~50-100ms (depends on data size)

## Development

### Local Testing
```bash
# Run tests
python -m pytest tests/

# Train and validate
python train_models.py

# Run app in debug mode
FLASK_ENV=development python app.py
```

### Code Structure

- `app.py`: Main Flask routes and logic
- `nlp_utils.py`: NLP models and text processing
- `next_word_predictor.py`: Word prediction model
- `db.py`: Database queries
- `train_models.py`: Model training and validation

## Future Improvements

- [ ] Add more training data for better accuracy
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add multilingual support
- [ ] Cache frequently accessed queries
- [ ] Add user authentication
- [ ] Implement real-time news feed
- [ ] Add sentiment analysis
- [ ] Improve UI/UX

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the `IMPROVEMENTS.md` file for recent changes
2. Review error logs: `python train_models.py`
3. Check database connection settings

## Contact

For deployment issues or bugs, check:
- Render dashboard logs
- Environment variables configuration
- Database connectivity

---

**Last Updated**: 2026-06-09
**Version**: 2.0 (Improved NLP Models)
