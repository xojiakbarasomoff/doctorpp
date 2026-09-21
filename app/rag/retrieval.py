from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.repositories.knowledge_base import KnowledgeBaseMatch, KnowledgeBaseRepository
from app.repositories.knowledge_document import ChunkMatch, KnowledgeChunkRepository

# Cosine distance below which a match is considered worth showing. This was
# 0.3, a starting heuristic that asked to be revisited once there were real
# queries to check it against. There are now, measured against the clinic's
# live knowledge base with scripts/probe_retrieval.py:
#
#   qon guruhi qancha              0.1712   the row it wants
#   jigar uzi narxi                0.2369   the row it wants
#   qon guruhini aniqlash          0.3190   the row it wants
#   ---
#   tish oldirsam qancha bo'ladi   0.3759   ear cleaning; the clinic has no
#                                           dentist, and this must stay out
#   mashina qanchaga sotiladi      0.4763   nothing to do with the clinic
#   salom qalaysiz                 0.5682   a greeting
#   futbol                         0.7361   noise
#
# The worst true match sits at 0.3190 and the nearest false one at 0.3759, so
# 0.35 falls in the gap between them rather than in the middle of either. At
# 0.3 the blood-group question was refused with its answer in the table; at
# 0.38 a patient asking about a tooth would be shown ear cleaning.
#
# What this is not is permission to quote whatever comes back. Everything
# admitted here is a near match, several of them at once, and rule 6 of the
# system prompt is what decides which of them -- if any -- names the service
# the patient actually asked for.
DEFAULT_MAX_DISTANCE = 0.35


async def retrieve_relevant_faqs(
    session: AsyncSession,
    query_text: str,
    embedding_provider: EmbeddingProvider | None = None,
    limit: int = 5,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
) -> list[KnowledgeBaseMatch]:
    """Embed query_text and return the current tenant's top matching FAQs,
    closest first. This is what the message pipeline will call to ground a
    reply — it does not generate an answer itself.

    max_distance drops matches whose cosine distance exceeds it (lower
    distance = more similar); pass None to skip filtering and always return
    up to `limit` matches regardless of how weak they are.
    """
    provider = embedding_provider or get_embedding_provider()
    [query_embedding] = await provider.embed([query_text])

    repo = KnowledgeBaseRepository(session)
    matches = await repo.search(query_embedding, limit=limit)

    if max_distance is None:
        return matches
    return [match for match in matches if match.distance <= max_distance]


# Cosine distance below which a piece of an uploaded file is worth showing.
#
# Not the FAQ threshold. A FAQ row is embedded by its question, so a patient's
# question lands near it; a chunk is a passage, and a short question is always
# further from the passage that answers it than from a question worded like
# it. DEFAULT_MAX_DISTANCE (0.35) would refuse nearly every true match.
#
# Measured with text-embedding-3-small against a price list, a preparation
# sheet and a doctor's timetable (three chunks, Uzbek, one Russian question,
# two typos), top-1 distance per message:
#
#   asking about the files      0.40 .. 0.70   "prostata uzi narx" 0.40,
#                                              "yakshanba ishlaysizmi" 0.64,
#                                              "uzi narhi" 0.61, "PSA qancha"
#                                              0.70, the Russian one 0.70
#   about the clinic, not them  0.59 .. 0.60   "manzilingiz qayerda" 0.59,
#                                              "qabulga yozilmoqchiman" 0.60
#   about something else        0.60 .. 0.84   "mashina narxi qancha" 0.64,
#                                              "kardiolog bormi" 0.73
#   chatter                     0.79 .. 0.84   "salom", "rahmat", "ok", a
#                                              telephone number
#
# The first two overlap, so no threshold separates them, and one that tried
# (0.60) lost "PSA qancha" and the Russian question. This sits above the
# last true match and below all the chatter: a piece that is shown and not
# needed costs a few tokens and the model ignores it; a piece that is not
# shown when it was needed is a patient told "I do not know" with the answer
# in the clinic's own file.
DEFAULT_CHUNK_MAX_DISTANCE = 0.72

# A few pieces, not a page: they share the prompt with the FAQ rows and the
# conversation.
DEFAULT_CHUNK_LIMIT = 4


async def retrieve_relevant_chunks(
    session: AsyncSession,
    query_embedding: list[float],
    limit: int = DEFAULT_CHUNK_LIMIT,
    max_distance: float | None = DEFAULT_CHUNK_MAX_DISTANCE,
) -> list[ChunkMatch]:
    """The current tenant's uploaded-file pieces nearest an embedded query."""
    matches = await KnowledgeChunkRepository(session).search(query_embedding, limit=limit)
    if max_distance is None:
        return matches
    return [match for match in matches if match.distance <= max_distance]


@dataclass(frozen=True)
class Knowledge:
    """Everything the clinic has written down that bears on one message."""

    faqs: list[KnowledgeBaseMatch] = field(default_factory=list)
    chunks: list[ChunkMatch] = field(default_factory=list)


async def retrieve_knowledge(
    session: AsyncSession,
    query_text: str,
    embedding_provider: EmbeddingProvider | None = None,
) -> Knowledge:
    """Question-and-answer rows and pieces of uploaded files, from one embedding.

    The message is embedded once and searched against both: two calls to the
    embedding service per patient message would be twice the latency and
    twice the bill for the same vector.
    """
    provider = embedding_provider or get_embedding_provider()
    [query_embedding] = await provider.embed([query_text])
    faqs = [
        match
        for match in await KnowledgeBaseRepository(session).search(query_embedding, limit=5)
        if match.distance <= DEFAULT_MAX_DISTANCE
    ]
    return Knowledge(
        faqs=faqs, chunks=await retrieve_relevant_chunks(session, query_embedding)
    )
