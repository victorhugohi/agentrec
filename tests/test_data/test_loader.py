"""Tests for the MovieLens data loader and preprocessing pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import DataSplits, MovieLensLoader

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_ratings() -> pd.DataFrame:
    """Synthetic ratings DataFrame with 4 users, 5 movies."""
    return pd.DataFrame(
        {
            "userId": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
            ],
            "movieId": [
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
                10,
                20,
                30,
                40,
                50,
            ],
            "rating": [
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
                4.5,
                3.5,
                2.5,
                1.5,
                0.5,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
                4.5,
                3.5,
                2.5,
                1.5,
                0.5,
                3.0,
                4.0,
                5.0,
                2.0,
                1.0,
                3.5,
                4.5,
                5.0,
                2.5,
                1.5,
                3.0,
                4.0,
                5.0,
                2.0,
                1.0,
                3.5,
                4.5,
                5.0,
                2.5,
                1.5,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                1.5,
                2.5,
                3.5,
                4.5,
                5.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                1.5,
                2.5,
                3.5,
                4.5,
                5.0,
            ],
            "timestamp": list(range(1, 21))
            + list(range(21, 41))
            + list(range(41, 46))
            + list(range(46, 66)),
        }
    )


@pytest.fixture
def sample_movies() -> pd.DataFrame:
    """Synthetic movies DataFrame."""
    return pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50],
            "title": [
                "Movie A (2000)",
                "Movie B (2001)",
                "Movie C (2002)",
                "Movie D (2003)",
                "Movie E (2004)",
            ],
            "genres": [
                "Action|Comedy",
                "Drama",
                "Sci-Fi|Thriller",
                "Romance",
                "Horror|Mystery",
            ],
        }
    )


@pytest.fixture
def tmp_data_dir() -> Path:
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def loader_with_data(
    tmp_data_dir: Path,
    sample_ratings: pd.DataFrame,
    sample_movies: pd.DataFrame,
) -> MovieLensLoader:
    """Loader with synthetic CSVs written to disk."""
    raw_dir = tmp_data_dir / "raw" / "ml-latest-small"
    raw_dir.mkdir(parents=True)
    sample_ratings.to_csv(raw_dir / "ratings.csv", index=False)
    sample_movies.to_csv(raw_dir / "movies.csv", index=False)
    return MovieLensLoader(variant="small", data_dir=tmp_data_dir)


# ── Tests ────────────────────────────────────────────────────────────


class TestMovieLensLoaderInit:
    """Test loader initialisation and validation."""

    def test_valid_variants(self) -> None:
        loader_sm = MovieLensLoader(variant="small")
        assert loader_sm.variant == "small"
        loader_25 = MovieLensLoader(variant="25m")
        assert loader_25.variant == "25m"

    def test_invalid_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="variant must be"):
            MovieLensLoader(variant="invalid")


class TestLoadRaw:
    """Test loading raw CSVs."""

    def test_load_ratings(self, loader_with_data: MovieLensLoader) -> None:
        df = loader_with_data.load_ratings()
        assert "userId" in df.columns
        assert "movieId" in df.columns
        assert "rating" in df.columns
        assert "timestamp" in df.columns
        assert len(df) == 65

    def test_load_movies(self, loader_with_data: MovieLensLoader) -> None:
        df = loader_with_data.load_movies()
        assert "movieId" in df.columns
        assert "title" in df.columns
        assert len(df) == 5


class TestPreprocess:
    """Test the full preprocessing pipeline."""

    def test_preprocess_returns_data_splits(self, loader_with_data: MovieLensLoader) -> None:
        splits = loader_with_data.preprocess(min_ratings=10)
        assert isinstance(splits, DataSplits)

    def test_filter_users_below_threshold(self, loader_with_data: MovieLensLoader) -> None:
        """User 3 has only 5 ratings and should be removed with min_ratings=10."""
        splits = loader_with_data.preprocess(min_ratings=10)
        # User 3 (original) only has 5 ratings → filtered out
        # Users 1, 2, 4 remain (20, 20, 20 ratings)
        assert splits.num_users == 3

    def test_ratings_normalised_to_zero_one(self, loader_with_data: MovieLensLoader) -> None:
        splits = loader_with_data.preprocess(min_ratings=5)
        for df in [splits.train, splits.val, splits.test]:
            if len(df) > 0:
                assert df["rating"].min() >= 0.0
                assert df["rating"].max() <= 1.0

    def test_ids_are_contiguous(self, loader_with_data: MovieLensLoader) -> None:
        splits = loader_with_data.preprocess(min_ratings=5)
        all_ratings = pd.concat([splits.train, splits.val, splits.test])
        user_ids = sorted(all_ratings["userId"].unique())
        movie_ids = sorted(all_ratings["movieId"].unique())
        assert user_ids == list(range(len(user_ids)))
        assert movie_ids == list(range(len(movie_ids)))

    def test_split_proportions(self, loader_with_data: MovieLensLoader) -> None:
        splits = loader_with_data.preprocess(min_ratings=5)
        total = len(splits.train) + len(splits.val) + len(splits.test)
        # With 4 users × 20 ratings (after filtering user 3 is kept with 5),
        # the total should match
        assert total > 0
        assert len(splits.train) >= len(splits.val)
        assert len(splits.train) >= len(splits.test)

    def test_split_by_time_order(self, loader_with_data: MovieLensLoader) -> None:
        """For each user, train timestamps < val timestamps < test timestamps."""
        splits = loader_with_data.preprocess(min_ratings=5)
        all_data = pd.concat(
            [
                splits.train.assign(split="train"),
                splits.val.assign(split="val"),
                splits.test.assign(split="test"),
            ]
        )
        for _uid, group in all_data.groupby("userId"):
            train_ts = group[group["split"] == "train"]["timestamp"]
            val_ts = group[group["split"] == "val"]["timestamp"]
            test_ts = group[group["split"] == "test"]["timestamp"]
            if len(train_ts) > 0 and len(val_ts) > 0:
                assert train_ts.max() <= val_ts.min()
            if len(val_ts) > 0 and len(test_ts) > 0:
                assert val_ts.max() <= test_ts.min()

    def test_movies_df_has_remapped_ids(self, loader_with_data: MovieLensLoader) -> None:
        splits = loader_with_data.preprocess(min_ratings=5)
        movie_ids = sorted(splits.movies["movieId"].unique())
        assert movie_ids == list(range(len(movie_ids)))

    def test_id_maps_cover_all_ids(self, loader_with_data: MovieLensLoader) -> None:
        splits = loader_with_data.preprocess(min_ratings=5)
        all_ratings = pd.concat([splits.train, splits.val, splits.test])
        assert set(all_ratings["userId"].unique()).issubset(set(splits.user_map.values()))
        assert set(all_ratings["movieId"].unique()).issubset(set(splits.movie_map.values()))


class TestSaveAndLoadProcessed:
    """Test Parquet serialisation round-trip."""

    def test_save_then_load(self, loader_with_data: MovieLensLoader) -> None:
        original = loader_with_data.preprocess(min_ratings=5)
        loaded = loader_with_data.load_processed()

        assert loaded is not None
        assert len(loaded.train) == len(original.train)
        assert len(loaded.val) == len(original.val)
        assert len(loaded.test) == len(original.test)
        assert loaded.user_map == original.user_map
        assert loaded.movie_map == original.movie_map

    def test_load_processed_returns_none_when_missing(self, tmp_data_dir: Path) -> None:
        loader = MovieLensLoader(variant="small", data_dir=tmp_data_dir)
        assert loader.load_processed() is None


class TestDataSplitsProperties:
    """Test DataSplits computed properties."""

    def test_num_users(self, loader_with_data: MovieLensLoader) -> None:
        splits = loader_with_data.preprocess(min_ratings=5)
        assert splits.num_users == len(splits.user_map)
        assert splits.num_users > 0

    def test_num_movies(self, loader_with_data: MovieLensLoader) -> None:
        splits = loader_with_data.preprocess(min_ratings=5)
        assert splits.num_movies == len(splits.movie_map)
        assert splits.num_movies > 0
