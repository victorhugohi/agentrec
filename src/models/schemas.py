"""Pydantic schemas for API request/response validation and agent I/O."""

from pydantic import BaseModel, Field

# --- API schemas ---


class RecommendationRequest(BaseModel):
    """Request body for getting recommendations.

    Attributes:
        user_id: The ID of the user to generate recommendations for.
        top_k: Number of recommendations to return.
    """

    user_id: int
    top_k: int = Field(default=10, ge=1, le=100)


class MovieItem(BaseModel):
    """A single movie item in a recommendation list.

    Attributes:
        item_id: Unique movie identifier.
        title: Movie title.
        score: Predicted relevance score.
        genres: List of genre tags.
    """

    item_id: int
    title: str = ""
    score: float = 0.0
    genres: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    """Response body containing a ranked list of recommendations.

    Attributes:
        user_id: The user these recommendations are for.
        recommendations: Ordered list of recommended movies.
        model: Name of the model used.
    """

    user_id: int
    recommendations: list[MovieItem]
    model: str = "ncf"


# --- Agent I/O schemas ---


class Rating(BaseModel):
    """A single user-item rating.

    Attributes:
        item_id: The rated movie's ID.
        rating: Rating value (1.0–5.0).
        title: Movie title.
        genres: Genre tags for the movie.
    """

    item_id: int
    rating: float
    title: str = ""
    genres: list[str] = Field(default_factory=list)


class UserProfileInput(BaseModel):
    """Input schema for the user profiler agent.

    Attributes:
        user_id: The user to build a profile for.
    """

    user_id: int


class UserProfileOutput(BaseModel):
    """Output schema for the user profiler agent.

    Attributes:
        user_id: The profiled user's ID.
        ratings: The user's rating history.
        embedding: Computed user embedding vector.
        rating_count: Total number of ratings.
        avg_rating: Mean rating value.
    """

    user_id: int
    ratings: list[Rating]
    embedding: list[float]
    rating_count: int
    avg_rating: float


class ContentAnalyzerInput(BaseModel):
    """Input schema for the content analyzer agent.

    Attributes:
        user_id: The user requesting recommendations (for candidate filtering).
        item_ids: Specific item IDs to analyze. If empty, returns all candidates.
    """

    user_id: int
    item_ids: list[int] = Field(default_factory=list)


class ContentItem(BaseModel):
    """Feature representation of a single movie.

    Attributes:
        item_id: Unique movie identifier.
        title: Movie title.
        genres: Genre tags.
        year: Release year.
        tags: Free-form user tags.
        embedding: Item feature embedding vector.
    """

    item_id: int
    title: str
    genres: list[str]
    year: int
    tags: list[str] = Field(default_factory=list)
    embedding: list[float]


class ContentAnalyzerOutput(BaseModel):
    """Output schema for the content analyzer agent.

    Attributes:
        user_id: The user this candidate set is for.
        items: List of analyzed content items with embeddings.
    """

    user_id: int
    items: list[ContentItem]


class ScoredItem(BaseModel):
    """An item with its predicted recommendation score.

    Attributes:
        item_id: Unique movie identifier.
        title: Movie title.
        score: Predicted relevance score from the NCF model.
        genres: Genre tags.
    """

    item_id: int
    title: str
    score: float
    genres: list[str] = Field(default_factory=list)


class RecsysEngineInput(BaseModel):
    """Input schema for the recommendation engine agent.

    Attributes:
        user_profile: Output from the user profiler agent.
        content_features: Output from the content analyzer agent.
        top_k: Number of recommendations to return.
    """

    user_profile: UserProfileOutput
    content_features: ContentAnalyzerOutput
    top_k: int = 10


class RecsysEngineOutput(BaseModel):
    """Output schema for the recommendation engine agent.

    Attributes:
        ranked_items: Items sorted by descending predicted score.
    """

    ranked_items: list[ScoredItem]


class UserProfile(BaseModel):
    """Representation of a user's preference profile.

    Attributes:
        user_id: Unique user identifier.
        rating_count: Total number of ratings by this user.
        avg_rating: Average rating given by this user.
        favorite_genres: Top genres by rating frequency.
    """

    user_id: int
    rating_count: int = 0
    avg_rating: float = 0.0
    favorite_genres: list[str] = Field(default_factory=list)


# --- Extended API schemas ---


class ErrorResponse(BaseModel):
    """Structured error response.

    Attributes:
        error: Error type identifier.
        message: Human-readable error description.
        request_id: Request trace ID for debugging.
    """

    error: str
    message: str
    request_id: str = ""


class UserHistoryResponse(BaseModel):
    """Response for user rating history.

    Attributes:
        user_id: The queried user.
        ratings: List of the user's ratings.
        rating_count: Total number of ratings.
        avg_rating: Mean rating value.
    """

    user_id: int
    ratings: list[Rating]
    rating_count: int
    avg_rating: float


class MovieSearchResult(BaseModel):
    """A single movie in search results.

    Attributes:
        movie_id: Unique movie identifier.
        title: Movie title.
        genres: Genre tags.
        year: Release year (0 if unknown).
    """

    movie_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    year: int = 0


class MovieSearchResponse(BaseModel):
    """Response for movie title search.

    Attributes:
        query: The original search query.
        results: Matching movies.
        total: Number of matches.
    """

    query: str
    results: list[MovieSearchResult]
    total: int


class AgentHealth(BaseModel):
    """Health status of a single agent.

    Attributes:
        name: Agent identifier.
        status: ``ok`` or ``degraded``.
        has_real_data: Whether the agent is backed by real data.
    """

    name: str
    status: str = "ok"
    has_real_data: bool = False


class HealthResponse(BaseModel):
    """Detailed health check response.

    Attributes:
        status: Overall status.
        model_loaded: Whether the NCF model is loaded.
        data_loaded: Whether processed data is available.
        num_users: Number of users in the dataset.
        num_movies: Number of movies in the dataset.
        agents: Per-agent health status.
    """

    status: str
    model_loaded: bool
    data_loaded: bool
    num_users: int = 0
    num_movies: int = 0
    agents: list[AgentHealth] = Field(default_factory=list)
