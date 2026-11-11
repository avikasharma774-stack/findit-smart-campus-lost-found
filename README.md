# FindIt — Smart Campus Lost & Found

A beginner-friendly prototype for a campus lost-and-found system. It compares lost and found item reports and ranks likely matches using text similarity.

## Run it

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually http://localhost:8501).

## What the prototype does

- Starts with a few sample lost/found reports.
- Lets a user add a new lost or found report.
- Lets the user select a report and search for likely matches.
- Ranks opposite-type reports by text similarity.
- Shows the similarity score and some shared description words.
- Lets the user view and download all reports as CSV.

## How the libraries are used

- **Pandas:** Stores the reports in a DataFrame, filters lost/found records, adds new rows, and exports CSV.
- **NumPy:** Sorts numerical similarity scores and finds their order.
- **Scikit-learn:** `TfidfVectorizer` converts text into numerical vectors; `cosine_similarity` compares those vectors.
- **Streamlit:** Creates the interactive web app.

## Explain the ML approach simply

1. Each report has an item name, category, description, and location.
2. `make_text()` combines those fields into one text string.
3. `TfidfVectorizer` gives words numerical weights. Words that help distinguish reports become useful features.
4. `cosine_similarity` compares the selected report with every report of the opposite type.
5. The scores are sorted from highest to lowest, and the top three are displayed.

## Important limitations (be honest about these)

- This is a prototype, not a production campus system.
- The sample data is illustrative.
- TF-IDF compares words, not image appearance or true meaning. It may miss matches when two descriptions use very different words.
- A high similarity score is not proof of ownership. A person should verify the item before returning it.
- Reports are stored in the app's session while it is running; this prototype does not use a permanent database or user login.

## Demo idea for the interview

1. Open **Find matches** and choose the sample lost black water bottle.
2. Show the found black steel bottle appearing as a likely match.
3. Go to **Add a report** and add a new lost item, for example: "red umbrella with curved wooden handle", location "auditorium".
4. Add a found report with a similar description, such as: "maroon umbrella with a brown curved handle", location "auditorium entrance".
5. Return to **Find matches** and demonstrate the new comparison.

## Files

- `app.py` — complete application
- `requirements.txt` — required Python libraries
