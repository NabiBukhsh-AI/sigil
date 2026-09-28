import pytest

from tests.helpers import clustered_embeddings


@pytest.fixture(scope="session")
def small_quantizer():
    from sigil_identifiers.quantizer import RQKMeans

    X = clustered_embeddings(3000)
    return RQKMeans(k=16, iters=15, restarts=1).fit(X), X
