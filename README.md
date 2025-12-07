## Review Name Finder: Automated Employee Recognition from Resident Reviews

[](https://www.python.org/)
[](https://streamlit.io/)
[](https://huggingface.co/dslim/bert-large-NER)

-----

## Overview

**Review Name Finder** is a Natural Language Processing (NLP) tool designed for the multifamily housing industry. It automatically parses online resident reviews to extract names of specific property staff members (such as leasing agents, maintenance supervisors, or managers).

By leveraging a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model, this tool allows researchers and property management companies to:

1.  **Identify Employees:** Automatically find and extract names mentioned in thousands of text reviews.
2.  **Correlate Sentiment:** Link specific employee names to the corresponding Star Rating given by the resident.
3.  **Automate Analysis:** Replace slow, manual review reading with an automated data pipeline for recognition and performance analysis.

This repository provides two implementations:

  * **`app.py`**: A user-friendly **Streamlit** web application.
  * **`script.py`**: A local Python script for batch processing.

-----

## Key Features

  * **Named Entity Recognition (NER):** Utilizes the powerful **`dslim/bert-large-NER`** model to accurately identify person entities (`PER`) within the unstructured text of reviews.
  * **Token Re-construction:** Includes custom post-processing logic to handle BERT's sub-word tokenization (e.g., merging split words like `Mar ##cus` back into `Marcus`).
  * **Star Rating Filters:** Allows users to filter reviews based on a minimum Star Rating threshold (e.g., only analyzing 4- and 5-star reviews).
  * **Aggregation:** The Streamlit version automatically groups the results by `PROPERTY NAME` and extracted `Name` to calculate the **Average Star Rating** and **Review Count** per employee.

-----

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/yourusername/review-name-finder.git
    cd review-name-finder
    ```

2.  Install the required dependencies using the provided `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### `requirements.txt` Content

```text
streamlit
pandas
transformers
torch
easygui
openpyxl
```

-----

## Data Requirements

The input file must contain the following specific column headers for the script to execute correctly.

| Column Name | Description |
| :--- | :--- |
| **`DATE`** | The date the review was submitted. |
| **`PROPERTY NAME`** | The name of the apartment community. |
| **`ACTUAL REVIEW`** | The full text content of the review (where names are extracted). |
| **`STAR RATING`** | The numerical rating (Integer 1-5). |
| **`REVIEWID`** | A unique identifier for the review (required for the Streamlit app). |

**Note:** The local script expects a **`.csv`** file named `input.csv`, while the Streamlit app is configured to handle uploaded **`.xlsx`** files.

-----

## Usage

### Option 1: Streamlit Web App (`app.py`)

This is the recommended method for an interactive, graphical workflow.

1.  Run the application from your terminal:
    ```bash
    streamlit run app.py
    ```
2.  Open the provided URL (typically `http://localhost:8501`) in your browser.
3.  Use the slider to set the minimum **Star Rating** threshold (e.g., setting it to `4` processes reviews with 4 or 5 stars).
4.  **Upload** your Excel file containing the required columns.
5.  The app will process the data and provide two download options:
      * **Raw output file (`raw.csv`):** The full list of reviews, with each extracted name on a separate row.
      * **Analysis file (`analysis.csv`):** The aggregated table showing **`PROPERTY NAME`**, **`Names`**, **`Average Star Rating`**, and **`Review Count`**.

### Option 2: Local Script (`namefinder.py`)

This is ideal for batch processing large datasets without a UI.

1.  Ensure your input data is saved as **`input.csv`** in the root directory.
2.  Run the script:
    ```bash
    python namefinder.py
    ```
3.  The script generates three output Excel files in the same directory:
      * **`premerge.xlsx`** (Intermediate data).
      * **`names.xlsx`** (A list of names and their corresponding review index).
      * **`output.xlsx`** (The final merged data with names attached to reviews).

-----

## Technical Explanation

The core of the application relies on **Named Entity Recognition (NER)**, a sequence labeling task. The BERT model processes the review text and assigns a label to each word (or sub-word token).

  * **`B-PER`** (Beginning of a Person): Marks the first token of a person's name.
  * **`I-PER`** (Inside of a Person): Marks subsequent tokens of a person's name.

The Python logic then uses these labels to stitch the tokens back together into a full name.

### Token Re-alignment

The most complex part of the code is handling sub-word tokenization. If the NER model splits a name (e.g., tokenizes "Marcus" as `Mar` and `##cus`), the code specifically looks for tokens starting with `##` and attempts to merge them with the preceding token to ensure names like "Don" and "ald" aren't missed or reported incorrectly.
