import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tweetopic import DMM, TopicPipeline

from app.utils.metadata import fetch_metadata

# Mapping of names to topic models
TOPIC_MODELS = {
    "nmf": NMF,
    "lda": LatentDirichletAllocation,
    "lsa": TruncatedSVD,
    "lsi": TruncatedSVD,
    "dmm": DMM,
}

# Mapping of names to vectorizers
VECTORIZERS = {
    "tf-idf": TfidfVectorizer,
    "bow": CountVectorizer,
}


def load_corpus() -> pd.DataFrame:
    return pd.read_csv("../dat/cleaned_corpus.csv")


def fit_pipeline(
    corpus: pd.DataFrame,
    vectorizer_name: str,
    min_df: int,
    max_df: float,
    model_name: str,
    n_topics: int,
) -> TopicPipeline:
    topic_model = TOPIC_MODELS[model_name](n_components=n_topics)
    vectorizer = VECTORIZERS[vectorizer_name](min_df=min_df, max_df=max_df)
    pipeline = TopicPipeline(vectorizer=vectorizer, topic_model=topic_model)
    pipeline = pipeline.fit(corpus.text)
    return pipeline


def calculate_importance(
    corpus: pd.DataFrame, pipeline: TopicPipeline
) -> pd.DataFrame:
    """Calculates for which groups in the corpus are the given topics specifically important.

    Parameters
    ----------
    corpus: DataFrame
        Data frame containing the cleaned corpus with ids.
    pipeline: TopicPipeline
        Fitted pipeline for topic modelling.

    Returns
    -------
    DataFrame
        Data about topic importances for each group.
    """
    # Selecting important works by joining with metadata
    # Only those get selected, which are assigned to a group
    md = fetch_metadata()
    md = md[["id_nummer", "group"]].dropna().set_index("id_nummer")
    # Removing unnecessary groups
    # NOTE: This will be removed in the future
    md = md[
        ~md.group.isin(
            [
                "Jewish Philosophy",
                "Jewish Pseudepigrapha",
                "NT Epistolography",
                "LXX Poetry/Wisdom",
            ]
        )
    ]
    important_works = corpus.join(md, how="inner")
    # Obtaining probabilities of each important working
    # belonging to a certain topic.
    probs = pipeline.transform(important_works.text)
    important_works = important_works.assign(
        # Assigning a topic embedding to each work
        topic=list(map(np.array, probs.tolist()))  # type: ignore
    )
    # Computing aggregated topic importances for each group
    importance = important_works.groupby("group").agg(
        topic=("topic", lambda s: np.stack(s).sum(axis=0))
    )
    # Normalizing these quantities, so that group sizes do
    # not mess with the analysis
    importance = importance.assign(
        topic=importance.topic.map(lambda a: a / a.max())
    )
    # Adding topic labels to the embeddings by enumerating them
    # and then exploding them
    importance = importance.applymap(
        lambda importances: list(enumerate(importances))
    ).explode("topic")
    # Splitting the tuples created by the enumeration to two columns
    importance[["topic", "importance"]] = importance.topic.tolist()
    # Resetting index, cause remember, group was the index because
    # of the aggregation
    importance = importance.reset_index()
    return importance


def calculate_top_words(
    pipeline: TopicPipeline, top_n: int = 30
) -> pd.DataFrame:
    """Arranges top N words of each topic to a DataFrame.

    Parameters
    ----------
    pipeline: TopicPipeline
        Fitted pipeline for topic modelling.
    top_n: int, default 30
        Number of words to include for each topic.

    Returns
    -------
    DataFrame
        Top N words for each topic with importance scores.

    Note
    ----
    Importance scores are exclusively calculated from the topic model's
    'components_' attribute and do not have anything to do with the empirical
    distribution of words in each topic. This has to be kept in mind, as some models
    keep counts in their 'components_' attribute (e.g. DMM).
    Would probably be smart to reconsider this in the future.
    """
    # Obtaining top N words for each topic
    top = pipeline.top_words(top_n=top_n)
    # Wrangling the data into tuple records
    records = []
    for i_topic, topic in enumerate(top):
        for word, importance in topic.items():
            records.append((i_topic, word, importance))
    # Adding to a dataframe
    top_words = pd.DataFrame(records, columns=["topic", "word", "importance"])
    # Normalizing word importances
    top_words = top_words.assign(
        importance=top_words.importance / top_words.importance.max()
    )
    return top_words
