"""Data loading, preparation, feature engineering, and splitting utilities."""

from __future__ import annotations
import json
from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter

DATASET_HANDLE = "tmdb/tmdb-movie-metadata"
MOVIES_FILENAME = "tmdb_5000_movies.csv"
CREDITS_FILENAME = "tmdb_5000_credits.csv"

MINIMUM_BUDGET = 0
FLOP_THRESHOLD = 2.5
MAX_TRACK_RECORD_RATIO = 10.0


def load_tmdb_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the TMDB movies and credits datasets from Kaggle."""
    movies = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_HANDLE,
        MOVIES_FILENAME,
    )
    credits = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_HANDLE,
        CREDITS_FILENAME,
    )
    return movies, credits


def _parse_json_list(value: object) -> list[dict]:
    """Safely parse a TMDB JSON-list column."""

    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def get_director(crew_json: object) -> str | None:
    """Extract the first listed director from a crew JSON value."""

    crew = _parse_json_list(crew_json)
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name")

    return None


def get_lead_actor(cast_json: object) -> str | None:
    """Extract the first listed actor from a cast JSON value."""

    cast = _parse_json_list(cast_json)
    if not cast:
        return None
    return cast[0].get("name")


def merge_movies_and_credits(
    movies: pd.DataFrame,
    credits: pd.DataFrame,
) -> pd.DataFrame:
    """Merge movie metadata with cast and crew data."""

    required_credit_columns = {"movie_id", "crew", "cast"}
    missing_columns = required_credit_columns - set(credits.columns)
    if missing_columns:
        raise ValueError(
            f"Credits data is missing columns: {sorted(missing_columns)}"
        )
    merged = movies.merge(
        credits[["movie_id", "crew", "cast"]],
        left_on="id",
        right_on="movie_id",
        how="inner",
        validate="one_to_one",
    )
    merged["director"] = merged["crew"].apply(get_director)
    merged["lead_actor"] = merged["cast"].apply(get_lead_actor)
    merged["release_date"] = pd.to_datetime(
        merged["release_date"],
        errors="coerce",
    )

    return merged


def add_financial_targets(
    df: pd.DataFrame,
    flop_threshold: float = FLOP_THRESHOLD,
) -> pd.DataFrame:
    """Add binary flop and continuous revenue-to-budget targets."""

    result = df.copy()
    valid_financials = (
        (result["budget"] > 0)
        & (result["revenue"] > 0)
    )
    result["rev_budget_ratio"] = np.nan
    result.loc[valid_financials, "rev_budget_ratio"] = (
        result.loc[valid_financials, "revenue"]
        / result.loc[valid_financials, "budget"]
    )
    result["flop"] = np.nan
    result.loc[valid_financials, "flop"] = (
        result.loc[valid_financials, "revenue"]
        < flop_threshold * result.loc[valid_financials, "budget"]
    ).astype(int)

    return result


def add_track_record_features(
    df: pd.DataFrame,
    maximum_ratio: float = MAX_TRACK_RECORD_RATIO,
) -> pd.DataFrame:
    """
    Add director and actor track-record features using prior films only.
    """

    result = df.copy()
    result = result.sort_values(
        ["release_date", "id"],
        na_position="last",
    ).reset_index(drop=True)
    result["rev_budget_ratio_capped"] = (
        result["rev_budget_ratio"].clip(upper=maximum_ratio)
    )
    result["director_success_ratio"] = (
        result.groupby("director")["rev_budget_ratio_capped"]
        .transform(lambda values: values.shift(1).expanding().mean())
    )
    result["actor_success_ratio"] = (
        result.groupby("lead_actor")["rev_budget_ratio_capped"]
        .transform(lambda values: values.shift(1).expanding().mean())
    )
    result["director_prior_count"] = (
        result.groupby("director")["rev_budget_ratio_capped"]
        .transform(lambda values: values.shift(1).expanding().count())
    )
    result["actor_prior_count"] = (
        result.groupby("lead_actor")["rev_budget_ratio_capped"]
        .transform(lambda values: values.shift(1).expanding().count())
    )
    return result


def construct_final_dataset(
    df: pd.DataFrame,
    minimum_budget: int = MINIMUM_BUDGET,
) -> pd.DataFrame:
    """Filter to valid high-budget films and fill in no track record features."""
    result = df[
        (df["budget"] > minimum_budget)
        & (df["revenue"] > 0)
        & df["release_date"].notna()
    ].copy()
    success_columns = [
        "director_success_ratio",
        "actor_success_ratio",
    ]
    count_columns = [
        "director_prior_count",
        "actor_prior_count",
    ]
    result[success_columns] = result[success_columns].fillna(1.0)
    result[count_columns] = result[count_columns].fillna(0)
    result["release_year"] = result["release_date"].dt.year.astype(int)
    result["flop"] = result["flop"].astype(int)
    return result.reset_index(drop=True)


def prepare_tmdb_dataset() -> pd.DataFrame:
    """Run the complete TMDB data-preparation pipeline."""
    movies, credits = load_tmdb_data()
    df = merge_movies_and_credits(movies, credits)
    df = add_financial_targets(df)
    df = add_track_record_features(df)
    df = add_pre_release_features(df)
    df = construct_final_dataset(df)
    return df


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the chronological train, validation, and test sets."""

    train = df[df["release_year"] < 2010].copy()
    validation = df[
        (df["release_year"] >= 2010)
        & (df["release_year"] <= 2013)
    ].copy()
    test = df[df["release_year"] >= 2014].copy()
    return train, validation, test


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: str | Path = "data/processed/movies_processed.csv",
) -> Path:
    """Save the processed dataset and return its output path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return output_path

def extract_names(json_value: object) -> list[str]:
    """Extract name fields from a TMDB JSON-list column."""
    items = _parse_json_list(json_value)
    return [
        item["name"]
        for item in items
        if isinstance(item, dict) and item.get("name")
    ]


def has_collection(value: object) -> int:
    """Return whether a movie belongs to a named collection."""
    if value is None:
        return 0
    if isinstance(value, float) and np.isnan(value):
        return 0
    if isinstance(value, str):
        value = value.strip().lower()
        return int(value not in {"", "nan", "none", "{}"})

    return 1

MAJOR_STUDIO_TERMS = [
    "warner bros",
    "universal",
    "paramount",
    "walt disney",
    "twentieth century fox",
    "20th century fox",
    "columbia pictures",
    "sony pictures",
    "metro-goldwyn-mayer",
    "new line cinema",
    "dreamworks",
]


def add_pre_release_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add genre, release timing, franchise, and studio features."""

    result = df.copy()

    # Release season
    month = result["release_date"].dt.month

    result["release_season"] = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        [
            "Winter",
            "Spring",
            "Summer",
            "Fall",
        ],
        default="Unknown",
    )

    # Sequel/franchise proxy derived from movie keywords
    franchise_terms = {
        "sequel",
        "prequel",
        "reboot",
        "remake",
        "spin off",
        "spin-off",
    }
    keyword_names = result["keywords"].apply(extract_names)
    result["is_sequel_or_remake"] = keyword_names.apply(
        lambda names: int(
            any(
                term in keyword.lower()
                for keyword in names
                for term in franchise_terms
            )
        )
    )

    # Production-company tier
    company_names = result["production_companies"].apply(
        extract_names
    )

    result["major_studio"] = company_names.apply(
        lambda names: int(
            any(
                studio_term in company.lower()
                for company in names
                for studio_term in MAJOR_STUDIO_TERMS
            )
        )
    )
    # Multi-hot genre columns
    genre_names = result["genres"].apply(extract_names)
    all_genres = sorted(
        {
            genre
            for movie_genres in genre_names
            for genre in movie_genres
        }
    )
    for genre in all_genres:
        column_name = (
            "genre_"
            + genre.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        result[column_name] = genre_names.apply(
            lambda movie_genres: int(genre in movie_genres)
        )

    return result