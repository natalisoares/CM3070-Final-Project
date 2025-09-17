# CM3070-Final-Project

The project follows the Natural Language Programming template, under the theme: Identifying research methodologies that are used in research in the computing disciplines.

The Research Classifier application helps classify research papers by:
- Discipline
- Subfield
- Methodology

It also provides:
- A detailed explanation of the classification
- Extracted datasets, tools, metrics, and environments
- An interactive AI assistant to explore the paper

The user interface of the application is clear, intuitive, and easy to navigate.

How to run:

1. Clone the repository or download the ZIP
2. Make sure you have Python 3.9 to 3.11 installed.
3. Install required packages:

   pip install streamlit openai scikit-learn spacy PyMuPDF python-docx joblib scipy numpy

   python -m spacy download en_core_web_sm

4. Run the app:

   streamlit run app.py

5. Upload PDF, DOCX or TXT research paper and click "Start Classification".

The app will display:
- A preview of your paper
- Tabs for Classification, Explanation, Data & Tools, Chat with AI Assistant

You can upload any research paper to try how it works.

Please note: the application includes a temporary OpenAI API key to allow reviewers to test the Explanation, AI Assistant, and Data & Tools features without setup. The key is rate-limited and will be automatically disabled if usage exceeds approximately 10,000 tokens, which covers 4-5 full classification requests. This safeguard is in place to prevent unauthorised usage. This key is intended strictly for academic assessment.
