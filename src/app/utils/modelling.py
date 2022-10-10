import numpy as np
import pandas as pd
from app.utils.metadata import fetch_metadata
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from tweetopic import DMM, TopicPipeline

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


def calculate_genre_importance(
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
    overall_importance = pipeline.topic_model.components_.sum(axis=0)
    vocab = pipeline.vectorizer.vocabulary_  # type: ignore
    # Wrangling the data into tuple records
    records = []
    for i_topic, topic in enumerate(top):
        for word, importance in topic.items():
            overall = overall_importance[vocab[word]]
            records.append((i_topic, word, importance, overall))
    # Adding to a dataframe
    top_words = pd.DataFrame(
        records, columns=["topic", "word", "importance", "overall_importance"]
    )
    return top_words


def calculate_topic_data(
    corpus: pd.DataFrame, pipeline: TopicPipeline
) -> pd.DataFrame:
    """Calculates topic positions in 2D space as well as topic sizes
    based on their empirical importance in the corpus.

    Parameters
    ----------
    corpus: DataFrame
        Data frame containing the cleaned corpus with ids.
    pipeline: TopicPipeline
        Fitted pipeline for topic modelling.

    Returns
    -------
    DataFrame
        Data about topic sizes and positions.
    """
    # Calculating topic predictions for each document.
    pred = pipeline.transform(corpus.text)
    _, n_topics = pred.shape  # type: ignore
    # Calculating topic size from the empirical importance of topics
    size = pred.sum(axis=0)  # type: ignore
    components = pipeline.topic_model.components_
    # Calculating topic positions with t-SNE
    x, y = (
        TSNE(perplexity=5, init="pca", learning_rate="auto")
        .fit_transform(components)
        .T
    )
    return pd.DataFrame(
        {"topic_id": range(n_topics), "x": x, "y": y, "size": size}
    )


def calculate_document_data(
    corpus: pd.DataFrame, pipeline: TopicPipeline
) -> pd.DataFrame:
    """Calculates document positions in 3D space as well as predictions for dominant topic.

    Parameters
    ----------
    corpus: DataFrame
        Data frame containing the cleaned corpus with ids.
    pipeline: TopicPipeline
        Fitted pipeline for topic modelling.

    Returns
    -------
    DataFrame
        Data about document positions and topic predictions + metadata.
    """
    # Calculating topic predictions for each document.
    pred = np.argmax(pipeline.transform(corpus.text), axis=1)  # type: ignore
    # Obtaining document-term matrix
    dtm = pipeline.vectorizer.transform(corpus.text)
    # Setting up dimensionality reduction pipeline
    dim_red_pipeline = Pipeline(
        [
            ("SVD", TruncatedSVD(50)),
            ("t-SNE", TSNE(3, init="random", learning_rate="auto")),
        ]
    )
    # Calculating dimensions in 3D space
    x, y, z = dim_red_pipeline.fit_transform(dtm).T
    documents = corpus.assign(x=x, y=y, z=z, topic_id=pred).drop(
        columns="text"
    )
    md = fetch_metadata()
    md = md[~md.skal_fjernes]
    md = md[["id_nummer", "værk", "forfatter", "group", "tlg_genre"]]
    documents = documents.merge(md, on="id_nummer", how="inner")
    documents = documents.assign(group=documents.group.fillna(""))
    return documents
