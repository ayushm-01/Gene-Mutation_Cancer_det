Project Overview
This repository contains a machine learning-based system for classifying gene mutations and detecting cancer markers. The application uses NLP techniques and classification models to analyze genetic sequences, identify mutations, and classify their potential role in cancer development.

Installation
# Clone the repository
git clone https://github.com/yourusername/gene-mutation-classification.git
cd gene-mutation-classification

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources (if needed)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

Usage
Running the Streamlit App
streamlit run app.py
