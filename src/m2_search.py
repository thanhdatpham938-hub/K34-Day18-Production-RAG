from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    segmented = word_tokenize(text, format="text")

    # underthesea nối từ ghép bằng "_"
    # Ví dụ: "nghỉ phép" -> "nghỉ_phép"
    # BM25 sẽ tokenize bằng split(), nên cần đổi "_" -> " "
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        # Lưu documents gốc
        self.documents = chunks

        # Tokenize từng chunk
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).split()
            for chunk in chunks
        ]

        # Xây dựng BM25 index
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K
    ) -> list[SearchResult]:
        """Search using BM25."""

        # Chưa index dữ liệu
        if self.bm25 is None:
            return []

        # Tokenize query
        tokenized_query = segment_vietnamese(query).split()

        # Tính BM25 score
        scores = self.bm25.get_scores(tokenized_query)

        # Sắp xếp document theo score giảm dần
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Chỉ lấy document có score > 0
        results = []

        for i in top_indices:
            if scores[i] <= 0:
                continue

            chunk = self.documents[i]

            results.append(
                SearchResult(
                    text=chunk["text"],
                    score=float(scores[i]),
                    metadata=chunk.get("metadata", {}),
                    method="bm25"
                )
            )

        return results

class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(
        self,
        chunks: list[dict],
        collection: str = COLLECTION_NAME
    ) -> None:
        """Index chunks into Qdrant."""

        from qdrant_client.models import Distance, VectorParams, PointStruct

        # 1. Tạo lại collection
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )

        # 2. Lấy text từ các chunks
        texts = [c["text"] for c in chunks]

        # 3. Encode text thành vector
        vectors = self._get_encoder().encode(
            texts,
            show_progress_bar=True
        )

        # 4. Tạo các Point để lưu vào Qdrant
        points = [
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    **chunk.get("metadata", {}),
                    "text": chunk["text"]
                }
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        # 5. Upload vectors vào Qdrant
        self.client.upsert(
            collection_name=collection,
            points=points
        )


    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME
    ) -> list[SearchResult]:
        """Search using dense vectors."""

        # 1. Encode query thành vector
        query_vector = self._get_encoder().encode(query).tolist()

        # 2. Search trong Qdrant
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k
        )

        # 3. Chuyển kết quả thành SearchResult
        return [
            SearchResult(
                text=pt.payload["text"],
                score=pt.score,
                metadata=pt.payload,
                method="dense"
            )
            for pt in response.points
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """
    Merge ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score:
        score(d) = Σ 1 / (k + rank)

    rank bắt đầu từ 1.
    """
    rrf_scores: dict[str, dict] = {}

    # Duyệt qua từng danh sách kết quả
    for result_list in results_list:

        # enumerate bắt đầu rank từ 0
        for rank, result in enumerate(result_list):

            # RRF rank bắt đầu từ 1
            actual_rank = rank + 1

            # Nếu document chưa xuất hiện
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "result": result,
                }

            # Cộng RRF score
            rrf_scores[result.text]["score"] += (
                1.0 / (k + actual_rank)
            )

    # Sắp xếp theo RRF score giảm dần
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    # Lấy top_k kết quả
    final_results = []

    for item in sorted_results[:top_k]:
        result = item["result"]

        # Cập nhật score và method
        result.score = item["score"]
        result.method = "hybrid"

        final_results.append(result)

    return final_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
