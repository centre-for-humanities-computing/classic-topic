"""Module for training topic pipelines and inferring data for plotting."""

import json
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as spr
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
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


def repeat_rows(df: pd.DataFrame, weight_col: str = "weight") -> pd.DataFrame:
    """Repeats rows by their individual weights in a specific weight column.

    Parameters
    ----------
    df: DataFrame
        Dataframe to repeat the rows of.
    weight_col: str, default 'weight'
        Name of the column that should be interpreted as weight.

    Returns
    -------
    DataFrame
        Dataframe with repeated rows.
    """
    index = df.index.to_series()
    new_index = []
    for ind, weight in zip(index, df[weight_col]):
        new_index.extend([ind] * weight)
    return df.loc[new_index]


MAX_FEATURES = 100_000


def prepare_corpus(
    corpus: pd.DataFrame, genre_weights: Dict[str, int]
) -> pd.DataFrame:
    """Prepares corpus for training with the given genre weights"""
    md = fetch_metadata()
    md = md[~md.skal_fjernes]
    md = md[["document_id", "work", "author", "group", "tlg_genre"]]
    md["group"] = md.group.fillna("Rest")
    corpus = corpus.merge(md, on="document_id", how="inner")
    corpus = corpus.assign(weight=corpus.group.map(genre_weights))
    # Weighting the different groups
    corpus = repeat_rows(corpus, weight_col="weight")
    return corpus


def fit_pipeline(
    corpus: pd.DataFrame,
    vectorizer_name: str,
    min_df: int,
    max_df: float,
    model_name: str,
    n_topics: int,
    n_gram_range: Tuple[int, int],
) -> TopicPipeline:
    """Fits topic pipeline with the given parameters.

    Parameters
    ----------
    corpus: DataFrame
        Corpus data containing texts and ids.
    vectorizer_name: {'tf-idf', 'bow'}
        Describes whether a TF-IDF of Bag of Words vectorizer should be fitted.
    min_df: int
        Minimum document frequency parameter of the vectorizer.
    max_df: float
        Minimum document frequency parameter of the vectorizer.
    model_name: {'nmf', 'lda', 'lsa'/'lsi', 'dmm'}
        Specifies which topic model should be trained on the corpus.
    n_topics: int
        Number of topics the model should find.
    n_gram_range: tuple of (int, int)
        N-gram range parameter of the vectorizer

    Returns
    -------
    TopicPipeline
        Fitted topic pipeline.
    """
    # Setting up pipeline
    topic_model = TOPIC_MODELS[model_name](n_components=n_topics)
    vectorizer = VECTORIZERS[vectorizer_name](
        min_df=min_df,
        max_df=max_df,
        ngram_range=n_gram_range,
        max_features=MAX_FEATURES,
    )
    pipeline = TopicPipeline(vectorizer=vectorizer, topic_model=topic_model)
    # Fitting the pipeline
    pipeline = pipeline.fit(corpus.text)
    return pipeline


def topic_document_importance(
    document_topic_matrix: np.ndarray,
    document_id: np.ndarray,
) -> Dict:
    """Calculates topic importances for each document."""
    coo = spr.coo_array(document_topic_matrix)
    topic_doc_imp = pd.DataFrame(
        dict(i_doc=coo.row, topic_id=coo.col, importance=coo.data)
    )
    return topic_doc_imp.to_dict()


def calculate_genre_importance(
    corpus: pd.DataFrame, pipeline: TopicPipeline
) -> pd.DataFrame:
    """Calculates for which groups in the corpus are the
    given topics specifically important.

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
    # Obtaining probabilities of each important working
    # belonging to a certain topic.
    probs = pipeline.transform(corpus.text)
    corpus = corpus.assign(
        # Assigning a topic embedding to each work
        topic=list(map(np.array, probs.tolist()))  # type: ignore
    )
    # Computing aggregated topic importances for each group
    importance = corpus.groupby("group").agg(
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


def word_relevance(
    topic_id: int,
    term_frequency: np.ndarray,
    topic_term_frequency: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Returns relevance scores for each topic for each word.

    Parameters
    ----------
    components: ndarray of shape (n_topics, n_vocab)
        Topic word probability matrix.
    alpha: float
        Weight parameter.

    Returns
    -------
    ndarray of shape (n_topics, n_vocab)
        Topic word relevance matrix.
    """
    probability = np.log(topic_term_frequency[topic_id])
    probability[probability == -np.inf] = np.nan
    lift = np.log(topic_term_frequency[topic_id] / term_frequency)
    lift[lift == -np.inf] = np.nan
    relevance = alpha * probability + (1 - alpha) * lift
    return relevance


def calculate_top_words(
    current_topic: int,
    top_n: int,
    alpha: float,
    term_frequency: np.ndarray,
    topic_term_frequency: np.ndarray,
    vocab: np.ndarray,
    **kwargs,
) -> pd.DataFrame:
    """Arranges top N words by relevance for the given topic into a DataFrame."""
    vocab = np.array(vocab)
    term_frequency = np.array(term_frequency)
    # topic_term_frequency = spr.csr_matrix(*topic_term_frequency)
    relevance = word_relevance(
        current_topic, term_frequency, topic_term_frequency, alpha=alpha
    )
    highest = np.argpartition(-relevance, top_n)[:top_n]
    res = pd.DataFrame(
        {
            "word": vocab[highest],
            "importance": topic_term_frequency[current_topic, highest],
            "overall_importance": term_frequency[highest],
            "relevance": relevance[highest],
        }
    )
    return res


def prepare_pipeline_data(vectorizer: Any, topic_model: Any) -> Dict:
    """Prepares data about the pipeline for storing
    in local store and plotting"""
    n_topics = topic_model.n_components
    vocab = vectorizer.get_feature_names_out()
    components = topic_model.components_
    # Making sure components are normalized
    # (remember this is not necessarily the case with some models)
    components = normalize(components, norm="l1", axis=1)
    return {
        "n_topics": n_topics,
        "vocab": vocab,
        "components": components,
    }


def prepare_transformed_data(
    vectorizer: Any, topic_model: Any, texts: Iterable[str]
) -> Dict:
    """Runs pipeline on the given texts and returns the document term matrix
    and the topic document distribution."""
    # Computing doc-term matrix for corpus
    document_term_matrix = vectorizer.transform(texts)
    # Transforming corpus with topic model for empirical topic data
    document_topic_matrix = topic_model.transform(document_term_matrix)
    return {
        "document_term_matrix": document_term_matrix,
        "document_topic_matrix": document_topic_matrix,
    }


def prepare_topic_data(
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    components: np.ndarray,
    n_topics: int,
    **kwargs,
) -> Dict:
    """Prepares data about topics for plotting."""
    components = np.array(components)
    # Calculating document lengths
    document_lengths = document_term_matrix.sum(axis=1)
    # Calculating an estimate of empirical topic frequencies
    topic_frequency = (document_topic_matrix.T * document_lengths).sum(axis=1)
    topic_frequency = np.squeeze(np.asarray(topic_frequency))
    # Calculating empirical estimate of term-topic frequencies
    # shape: (n_topics, n_vocab)
    topic_term_frequency = (components.T * topic_frequency).T
    # Empirical term frequency
    term_frequency = topic_term_frequency.sum(axis=0)
    term_frequency = np.squeeze(np.asarray(term_frequency))
    # Determining topic positions with TSNE
    topic_pos = (
        TSNE(perplexity=5, init="pca", learning_rate="auto")
        .fit_transform(components)
        .T
    )
    topic_id = np.arange(n_topics)
    return {
        "topic_id": topic_id,
        "topic_frequency": topic_frequency,
        "topic_pos": topic_pos,
        "term_frequency": term_frequency,
        "topic_term_frequency": topic_term_frequency,
    }


def prepare_document_data(
    corpus: pd.DataFrame,
    document_term_matrix: np.ndarray,
    document_topic_matrix: np.ndarray,
    **kwargs,
) -> Dict:
    dominant_topic = np.argmax(document_topic_matrix, axis=1)
    # Setting up dimensionality reduction pipeline
    dim_red_pipeline = Pipeline(
        [
            ("SVD", TruncatedSVD(50)),
            ("t-SNE", TSNE(3, init="random", learning_rate="auto")),
        ]
    )
    # Calculating positions in 3D space
    x, y, z = dim_red_pipeline.fit_transform(document_term_matrix).T
    documents = corpus.assign(
        x=x,
        y=y,
        z=z,
        i_doc=np.arange(len(corpus.index)),
        topic_id=dominant_topic,
    )
    importance_sparse = topic_document_importance(
        document_topic_matrix,
        document_id=documents.document_id,  # type: ignore
    )
    return {
        "document_data": documents.to_dict(),
        "document_topic_importance": importance_sparse,
    }


def serialize_save_data(fit_data: Dict, topic_names: List[str]) -> str:
    """Serializes model fit and topic name data and turns it into JSON,
    so that it can be saved.

    Parameters
    ----------
    fit_data: dict
        Data about the model fit.
    topic_names: list of str
        Data about the topic names.
    """
    return json.dumps(
        {"fit_data": {**fit_data, "loaded": True}, "topic_names": topic_names}
    )
