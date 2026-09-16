from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.repositories.knowledge_base import KnowledgeBaseMatch, KnowledgeBaseRepository

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
