from types import SimpleNamespace

from memory import Memory


def test_generate_embedding_with_models_embed_content():
    mem = Memory.__new__(Memory)

    class ModelsAPI:
        def embed_content(self, model, contents):
            return {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}

    mem.client = SimpleNamespace(models=ModelsAPI())

    emb = mem._generate_embedding("hello")

    assert emb == [0.1, 0.2, 0.3]


def test_generate_embedding_falls_back_to_embeddings_create():
    mem = Memory.__new__(Memory)

    class EmbeddingsAPI:
        def create(self, model, input):
            return {"embedding": [0.4, 0.5]}

    mem.client = SimpleNamespace(embeddings=EmbeddingsAPI())

    emb = mem._generate_embedding("world")

    assert emb == [0.4, 0.5]
