import pandas as pd
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textstat import flesch_kincaid_grade
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.text import Text
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download("stopwords")
nltk.download("punkt_tab")


def unique_word_ratio(statement):
    words = statement.split()
    return len(set(words)) / len(words) if len(words) > 0 else 0


def categorize_instruction(value):
    value_lower = value.lower()
    if "democrat" in value_lower:
        return "democrat"
    elif "republican" in value_lower:
        return "republican"
    else:
        return ""


def is_truth(party, instruction):
    return party.lower() == instruction.lower()


def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def get_word_frequencies(party, instruction, df):
    stop_words = set(stopwords.words("english"))  # Set of English stopwords
    words = (
        " ".join(
            df[(df["party"] == party) & (df["instruction"] == instruction)]["statement"]
        )
        .lower()
        .split()
    )
    filtered_words = [
        word for word in words if word not in stop_words and word.isalpha()
    ]
    return Counter(filtered_words).most_common(10)


# Load and process data
input_file_path = "writers.xlsx"
output_file_path = "writers_edited.xlsx"


# Main processing function
def process_and_save_excel_file(input_file_path, output_file_path):
    df = pd.read_excel(input_file_path)
    columns_to_keep = [
        "party",
        "age_1",
        "d_topic",
        "r_topic",
        "d_statement",
        "r_statement",
        "gender",
        "race",
        "language",
        "education",
        "Off Topic",
        "Bad Language",
    ]
    trimmed_df = df[columns_to_keep]
    trimmed_df["statement"] = trimmed_df["d_statement"].fillna("") + trimmed_df[
        "r_statement"
    ].fillna("")
    trimmed_df["statement_length"] = trimmed_df["statement"].str.strip().str.len()
    trimmed_df["instruction"] = trimmed_df["d_topic"].fillna("") + trimmed_df[
        "r_topic"
    ].fillna("")
    trimmed_df = trimmed_df[trimmed_df["statement"].str.strip() != ""]
    trimmed_df = trimmed_df[trimmed_df["instruction"].str.strip() != ""]
    trimmed_df["party"] = trimmed_df["party"].apply(lambda x: x.lower())
    trimmed_df["instruction"] = trimmed_df["instruction"].apply(categorize_instruction)
    trimmed_df["truth"] = trimmed_df.apply(
        lambda row: is_truth(row["party"], row["instruction"]), axis=1
    )
    trimmed_df.to_excel(output_file_path, index=False)
    return trimmed_df


fixed_df = process_and_save_excel_file(input_file_path, output_file_path)

fixed_df["sentiment"] = fixed_df["statement"].apply(get_sentiment)
avg_sentiment_by_instruction = (
    fixed_df.groupby(["party", "instruction"])["sentiment"].mean().unstack()
)
plt.figure(figsize=(10, 6))
sns.heatmap(avg_sentiment_by_instruction, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Sentiment Analysis by Party and Instruction")
plt.xlabel("Assigned Instruction")
plt.ylabel("Actual Party")
plt.show()

fixed_df["vocab_diversity"] = fixed_df["statement"].apply(unique_word_ratio)
avg_vocab_by_instruction = (
    fixed_df.groupby(["party", "instruction"])["vocab_diversity"].mean().unstack()
)
plt.figure(figsize=(10, 6))
sns.heatmap(avg_vocab_by_instruction, annot=True, cmap="viridis", fmt=".2f")
plt.title("Vocabulary Diversity by Party and Instruction")
plt.xlabel("Assigned Instruction")
plt.ylabel("Actual Party")
plt.show()

top_words_dem_as_rep = get_word_frequencies("democrat", "republican", fixed_df)
top_words_rep_as_dem = get_word_frequencies("republican", "democrat", fixed_df)
wordcloud_dem_as_rep = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(dict(top_words_dem_as_rep))
wordcloud_rep_as_dem = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(dict(top_words_rep_as_dem))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_dem_as_rep, interpolation="bilinear")
plt.axis("off")
plt.title("Words Used by Democrats Writing as Republicans")
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_rep_as_dem, interpolation="bilinear")
plt.axis("off")
plt.title("Words Used by Republicans Writing as Democrats")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(
    data=fixed_df, x="sentiment", hue="instruction", multiple="stack", kde=True, bins=20
)
plt.title("Sentiment Distribution by Instruction")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.show()
avg_length_by_instruction = (
    fixed_df.groupby(["party", "instruction"])["statement_length"].mean().unstack()
)
avg_length_by_instruction.plot(kind="bar", stacked=False, figsize=(10, 6))
plt.title("Average Statement Length by Actual and Assigned Party")
plt.xlabel("Actual Party")
plt.ylabel("Average Statement Length")
plt.show()
from sklearn.feature_extraction.text import CountVectorizer


def plot_word_heatmap(df, party, instruction):
    vectorizer = CountVectorizer(max_features=10, stop_words="english")
    X = vectorizer.fit_transform(
        df[(df["party"] == party) & (df["instruction"] == instruction)]["statement"]
    )
    words = vectorizer.get_feature_names_out()
    word_counts = X.toarray().sum(axis=0)
    sns.heatmap(
        [word_counts], annot=True, xticklabels=words, cmap="coolwarm", cbar=False
    )
    plt.title(f"Most Common Words by {party} Writing as {instruction}")
    plt.show()


plot_word_heatmap(fixed_df, "democrat", "republican")
plot_word_heatmap(fixed_df, "republican", "democrat")
plt.figure(figsize=(10, 6))
sns.boxplot(data=fixed_df, x="instruction", y="sentiment", hue="party")
plt.title("Sentiment Distribution by Party and Instruction")
plt.xlabel("Assigned Instruction")
plt.ylabel("Sentiment Score")
plt.show()
truthful_text = " ".join(fixed_df[fixed_df["truth"] == True]["statement"])
non_truthful_text = " ".join(fixed_df[fixed_df["truth"] == False]["statement"])

wordcloud_truthful = WordCloud(
    width=800, height=400, background_color="white"
).generate(truthful_text)
wordcloud_non_truthful = WordCloud(
    width=800, height=400, background_color="white"
).generate(non_truthful_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_truthful, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for Truthful Statements")
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_non_truthful, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for Non-Truthful Statements")
plt.show()
avg_sentiment_truth = fixed_df.groupby("truth")["sentiment"].mean()
avg_sentiment_truth.plot(kind="bar", color=["green", "red"])
plt.title("Average Sentiment for Truthful vs Non-Truthful Statements")
plt.ylabel("Average Sentiment Score")
plt.show()
avg_vocab_by_truth = fixed_df.groupby("truth")["vocab_diversity"].mean()
avg_vocab_by_truth.plot(kind="bar", color=["green", "red"])
plt.title("Vocabulary Diversity for Truthful vs Non-Truthful Statements")
plt.ylabel("Average Vocabulary Diversity")
plt.show()


def get_top_bigrams(df, party, instruction):
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words="english")
    # Filter the DataFrame based on party and instruction and generate bigrams
    statements = df[(df["party"] == party) & (df["instruction"] == instruction)][
        "statement"
    ]
    X = vectorizer.fit_transform(statements)
    feature_names = vectorizer.get_feature_names_out()

    # Normalize bigrams by sorting the words within each bigram alphabetically
    sorted_bigrams = [" ".join(sorted(bigram.split())) for bigram in feature_names]

    # Count the occurrences of each bigram
    counts = X.toarray().sum(axis=0)
    bigram_counts = dict(zip(sorted_bigrams, counts))

    deduped_counts = Counter()
    for bigram, count in bigram_counts.items():
        deduped_counts[bigram] += count

    # Convert the deduped_counts to a DataFrame and sort by count
    result_df = pd.DataFrame(list(deduped_counts.items()), columns=["bigram", "count"])
    result_df = result_df.sort_values(by="count", ascending=False).head(10)

    return result_df


dem_as_rep_bigrams = get_top_bigrams(fixed_df, "democrat", "republican")
rep_as_dem_bigrams = get_top_bigrams(fixed_df, "republican", "democrat")
dem_as_dem_bigrams = get_top_bigrams(fixed_df, "democrat", "democrat")
rep_as_rep_bigrams = get_top_bigrams(fixed_df, "republican", "republican")
plt.figure(figsize=(10, 8))
plt.barh(dem_as_rep_bigrams["bigram"], dem_as_rep_bigrams["count"], color="skyblue")
plt.xlabel("Count")
plt.ylabel("Bigram")
plt.title("Top Bigrams Used by Democrats Writing as Republicans")
plt.gca().invert_yaxis()
plt.show()
plt.figure(figsize=(10, 8))
plt.barh(rep_as_dem_bigrams["bigram"], rep_as_dem_bigrams["count"], color="skyblue")
plt.xlabel("Count")
plt.ylabel("Bigram")
plt.title("Top Bigrams Used by Republicans Writing as Democrats")
plt.gca().invert_yaxis()
plt.show()
plt.figure(figsize=(10, 8))
plt.barh(rep_as_rep_bigrams["bigram"], rep_as_rep_bigrams["count"], color="skyblue")
plt.xlabel("Count")
plt.ylabel("Bigram")
plt.title("Top Bigrams Used by Republicans Writing as Republicans")
plt.gca().invert_yaxis()
plt.show()
plt.figure(figsize=(10, 8))
plt.barh(dem_as_dem_bigrams["bigram"], dem_as_dem_bigrams["count"], color="skyblue")
plt.xlabel("Count")
plt.ylabel("Bigram")
plt.title("Top Bigrams Used by Democrats Writing as Democrats")
plt.gca().invert_yaxis()
plt.show()


avg_length_by_truth = fixed_df.groupby("truth")["statement_length"].mean()
avg_length_by_truth.plot(kind="bar", color=["blue", "orange"])
plt.title("Average Statement Length by Truth")
plt.ylabel("Average Statement Length")
plt.show()

party_by_race = (
    pd.crosstab(fixed_df["race"], fixed_df["party"], normalize="index") * 100
)
party_by_race.plot(kind="bar", stacked=True, color=["green", "red"])
plt.title("Party by Race")
plt.ylabel("Percentage")
plt.show()
age_bins = pd.cut(
    fixed_df["age_1"],
    bins=[18, 30, 45, 60, 100],
    labels=["18-30", "31-45", "46-60", "60+"],
)
truth_by_age_party = (
    pd.crosstab(age_bins, fixed_df["party"], values=fixed_df["truth"], aggfunc="mean")
    * 100
)
sns.heatmap(truth_by_age_party, annot=True, cmap="Blues")
plt.title("Truthfulness by Age and Party")
plt.xlabel("Party")
plt.ylabel("Age Group")
plt.show()

fixed_df["readability"] = fixed_df["statement"].apply(flesch_kincaid_grade)
avg_readability_by_instruction = (
    fixed_df.groupby(["party", "instruction"])["readability"].mean().unstack()
)
sns.heatmap(avg_readability_by_instruction, annot=True, cmap="YlGnBu")
plt.title("Average Readability of Statements by Party and Instruction")
plt.show()
corr_matrix = fixed_df[
    ["sentiment", "truth", "vocab_diversity", "statement_length"]
].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Sentiment, Truth, Diversity, and Length")
plt.show()


def perform_topic_modeling_and_plot(df, party, instruction, n_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    statements = df[(df["party"] == party) & (df["instruction"] == instruction)][
        "statement"
    ]
    X = vectorizer.fit_transform(statements)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx}: {', '.join(top_words)}")

    print(f"Topics for {party} writing as {instruction}:")
    for topic in topics:
        print(topic)


for party in ["democrat", "republican"]:
    for instruction in ["democrat", "republican"]:
        perform_topic_modeling_and_plot(fixed_df, party, instruction)
plt.figure(figsize=(12, 6))
sns.boxplot(data=fixed_df, x="instruction", y="vocab_diversity", hue="party")
plt.title("Lexical Diversity by Party and Instruction")
plt.xlabel("Assigned Instruction")
plt.ylabel("Lexical Diversity")
plt.show()


def find_collocations(text):
    words = nltk.word_tokenize(text.lower())
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)

    # Only consider bigrams that appear more than 2 times
    finder.apply_freq_filter(3)

    # Get top 10 collocations by likelihood ratio
    collocations = finder.nbest(bigram_measures.likelihood_ratio, 10)
    return collocations


def plot_collocations(df, party, instruction):
    all_statements = " ".join(
        df[(df["party"] == party) & (df["instruction"] == instruction)]["statement"]
    )
    collocations = find_collocations(all_statements)
    print(f"Top 10 Collocations for {party} writing as {instruction}: {collocations}")


for party in ["democrat", "republican"]:
    for instruction in ["democrat", "republican"]:
        plot_collocations(fixed_df, party, instruction)


def concordance_analysis(word, text):
    tokens = nltk.word_tokenize(text.lower())
    text_obj = Text(tokens)

    print(f"Concordance for '{word}':")
    text_obj.concordance(word, width=80, lines=10)


def concordance_analysis_by_party_instruction(df, word):
    for party in ["democrat", "republican"]:
        for instruction in ["democrat", "republican"]:
            all_statements = " ".join(
                df[(df["party"] == party) & (df["instruction"] == instruction)][
                    "statement"
                ]
            )
            print(f"Concordance for '{word}' - {party} writing as {instruction}:")
            concordance_analysis(word, all_statements)


concordance_analysis_by_party_instruction(fixed_df, "freedom")


def co_occurrence_matrix(text, window_size=2):
    vectorizer = CountVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(text)
    words = vectorizer.get_feature_names_out()

    # Convert to dense matrix and sum co-occurrences within the window
    co_occurrence_matrix = X.T @ X
    co_occurrence_df = pd.DataFrame(
        co_occurrence_matrix.toarray(), index=words, columns=words
    )

    return co_occurrence_df


co_occurrence_df = co_occurrence_matrix(fixed_df["statement"])
sns.heatmap(co_occurrence_df, cmap="Blues")
plt.title("Co-occurrence Matrix")
plt.show()
