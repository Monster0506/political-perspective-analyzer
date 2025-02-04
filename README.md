# Ideological Turing Test Analysis

## Overview

This project analyzes data from the **Ideological Turing Test** (ITT), where participants write statements as either their true political affiliation (Democrat or Republican) or pretend to be from the opposing party. The goal of this project is to explore how people express their political beliefs, whether others can tell the difference between real and feigned statements, and the linguistic features that distinguish these different types of statements.

Original Data Source:

- [Ideological Turing Test Data](https://osf.io/45w68)
- File used: `ITT data and code_upload/writers.csv`

Background:  
Inspired by [this article](https://www.experimental-history.com/p/ideological-turing-test), the data includes party affiliation, statement content, assigned task (Democrat or Republican), and other demographic details.

## Project Objective

The primary objective is to visualize and analyze the similarities and differences between Republicans and Democrats writing as themselves versus when they attempt to adopt the perspective of the opposing party.

## Dataset Processing

We processed the `writers.csv` file to include the following key variables:

- **Party**: The actual party of the writer (Democrat/Republican).
- **Instruction**: Whether the writer is writing as a Democrat or Republican.
- **Truth**: Whether the writer's instruction aligns with their real party (True/False).
- **Statement**: The written statement, compiled from available Democrat and Republican statements.
- **Sentiment**: The polarity of each statement, calculated using `TextBlob`.
- **Vocabulary Diversity**: The ratio of unique words in each statement.
- **Statement Length**: The length of each statement, measured in characters.
- **Readability**: The reading difficulty of each statement, measured by Flesch-Kincaid Grade Level.

## Code Explanation

### Libraries Used

- **pandas**: Data manipulation and processing.
- **nltk**: Natural Language Toolkit for text analysis (stopword removal, tokenization, collocations).
- **textstat**: Readability measurement.
- **TextBlob**: Sentiment analysis.
- **scikit-learn**: Vectorization, topic modeling, and Latent Dirichlet Allocation (LDA).
- **matplotlib** & **seaborn**: Data visualization (heatmaps, bar charts, word clouds).
- **wordcloud**: Generates visual word clouds.

### Key Functions

1. **Data Processing**:

   - `process_and_save_excel_file(input_file_path, output_file_path)`: Reads and processes the raw dataset, cleaning columns, generating truth labels, and saving the processed file.

2. **Sentiment and Diversity Analysis**:

   - `get_sentiment(text)`: Returns sentiment polarity of a statement.
   - `unique_word_ratio(statement)`: Returns the vocabulary diversity of a statement.

3. **Word Frequency and Bigram Analysis**:

   - `get_word_frequencies(party, instruction, df)`: Returns the most common words for given party and instruction.
   - `get_top_bigrams(df, party, instruction)`: Identifies the top 10 bigrams (word pairs) used in the statements.

4. **Topic Modeling**:

   - `perform_topic_modeling_and_plot(df, party, instruction, n_topics=3)`: Uses LDA to identify topics from statements written by each party, under each instruction.

5. **Collocation and Concordance Analysis**:

   - `find_collocations(text)`: Finds common word pairings (bigrams) based on frequency.
   - `concordance_analysis(word, text)`: Shows concordance for a specific word, visualizing its context in the statements.

6. **Visualizations**:
   - **Heatmaps**: Used to show average sentiment, vocabulary diversity, and readability across party and instruction.
   - **Word Clouds**: Visual representations of the most frequently used words by party/instruction.
   - **Box Plots and Bar Charts**: Visualize distributions of sentiment, statement length, and lexical diversity.

## Instructions to Run

1. Set up a virtual environment and install the necessary Python libraries using:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Ensure NLTK stopwords and other required corpora are downloaded:

   ```python
   import nltk
   nltk.download("stopwords")
   nltk.download("punkt")
   ```

3. Place the original `writers.csv` file in the same directory as the script.
4. Convert the `writers.csv` file into an Excel (`.xlsx`) file using Excel, or change the code to handle a `csv`.

5. Run the script to process the data and generate visualizations:

   ```bash
   python main.py
   ```

The script will produce visual outputs like sentiment heatmaps, word clouds, statement length comparisons, and co-occurrence matrices.

## Visual Output Examples

1. **Sentiment Analysis Heatmap**: Compares sentiment scores for statements written as Democrats or Republicans.
2. **Word Clouds**: Shows commonly used words by Democrats writing as Republicans and vice versa.
3. **Topic Modeling**: Displays the primary topics discussed by each party when adopting their own or the opposing perspective.
4. **Statement Length and Lexical Diversity**: Bar charts showing the average length and vocabulary richness of statements by truthfulness.

## Key Findings

- TODO

## Future Directions

Further analysis could incorporate machine learning techniques for automated party classification based on language features, or expand the dataset to examine more nuanced linguistic traits like sarcasm or emotional tone.
