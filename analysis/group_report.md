# Group Report — Lab 18: Production RAG

**Học viên:** Phạm Thanh Đạt (bài tập cá nhân — làm đủ cả 5 module)
**Ngày:** 2026-08-19

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Phạm Thanh Đạt | M1: Chunking | ☑ | 13/13 |
| Phạm Thanh Đạt | M2: Hybrid Search | ☑ | 5/5 |
| Phạm Thanh Đạt | M3: Reranking | ☑ | 5/5 |
| Phạm Thanh Đạt | M4: Evaluation | ☑ | 4/4 |
| Phạm Thanh Đạt | M5: Enrichment | ☑ | 10/10 |

`pytest tests/ -v`: **37/37 passed** (test_m1: 13, test_m2: 5, test_m3: 5, test_m4: 4, test_m5: 10).

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8319 | 0.7225 | **-0.1094** |
| Answer Relevancy | 0.6854 | 0.6834 | -0.0020 |
| Context Precision | 0.9250 | 0.9500 | +0.0250 |
| Context Recall | 0.9083 | 0.7917 | **-0.1167** |

Chạy thật `python main.py` (OpenAI key `sk-proj-...`, model `gpt-4o-mini`, 20 câu hỏi `test_set.json`). Trước khi có key đúng, đã thử 2 lần: (1) key rỗng cố ý — validate pipeline chạy sạch $0, cả 4 metric ra 0.0 đúng theo try/except; (2) key OpenRouter nhầm định dạng — mọi call LLM fail 401 (không tốn phí vì fail trước khi model chạy), nhưng nhờ đó phát hiện và fix 1 bug thật trong `main.py` (`os.rename` crash khi report file đích đã tồn tại → đổi `os.replace`).

## Chunking Stats (thật, từ `compare_strategies` trên 26 documents/`data/`)

| Strategy | Count | Avg length (chars) |
|----------|-------|---------------------|
| Basic (paragraph) | 51 | 410 |
| Semantic (threshold 0.85) | 208 | 99 |
| Hierarchical | 11 parents / 87 children | 1907 / 239 |
| Structure-aware | 106 (95 unique sections) | — |

## Key Findings

1. **Biggest improvement:** Context precision (+0.025, 0.95 vs 0.925) — hybrid search (BM25 + dense qua RRF) + cross-encoder rerank lọc nhiễu tốt hơn dense-only.
2. **Biggest challenge:** Không phải RAGAS số xấu do bug logic — mà là **production pipeline thua naive baseline ở faithfulness (-0.109) và context_recall (-0.117)**, ngược với kỳ vọng "production luôn tốt hơn baseline". Bottom-5 trong `failure_analysis.md` cho thấy 3/5 case retrieval đúng nhưng generation (gpt-4o-mini, prompt đơn giản) vẫn trả lời sai — chai cổ chai không nằm ở M1-M3 mà ở prompt cuối cùng cho LLM.
3. **Surprise finding:** Case #2 trong Bottom-5 (mua laptop 30 triệu) bị **keyword collision theo số liệu** — reranker kéo nhầm 1 document hoàn toàn khác chủ đề (chính sách hoàn chi đào tạo) vào top-3 chỉ vì cùng chứa con số "30.000.000 VNĐ", chiếm chỗ của document đúng (quy trình mua sắm). Đây là loại lỗi mà cross-encoder rerank thuần semantic không tự sửa được.

Thêm 1 finding về môi trường (không phải RAGAS nhưng tốn nhiều thời gian nhất): 5 vấn đề setup liên tiếp — Python 3.13 thiếu wheel `numpy<2`, OneDrive khóa file giữa lúc `pip install`, Windows console `cp1258` crash khi print emoji, `main.py` bug `os.rename`, và API key nhầm loại (OpenRouter vs OpenAI) — chi tiết ở `analysis/reflections/reflection_PhamThanhDat.md`.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** Production thắng context_precision (+0.025) nhưng thua faithfulness (-0.109) và context_recall (-0.117) — kết quả ngược kỳ vọng, đáng để đào sâu hơn là chỉ báo cáo "production tốt hơn".
2. **Biggest win — module nào, tại sao:** M2+M3 (Hybrid search + rerank) — context_precision cao nhất trong cả 2 pipeline (0.95), cho thấy khi retrieval đúng thì rerank lọc nhiễu hiệu quả.
3. **Case study — 1 failure, Error Tree:** Case #3 trong `failure_analysis.md` ("thử việc có nghỉ phép năm không") — context retrieval đúng 100% (score 0.995) nhưng faithfulness vẫn = 0.0 → root cause nằm ở generation prompt, không phải M1/M2/M3.
4. **Next optimization nếu có thêm 1 giờ:** Tighten generation prompt (few-shot cho câu hỏi cần tính toán/version-conflict) — vì 3/5 Bottom-5 fail ở generation dù context đúng, đây là đòn bẩy cao nhất hiện tại, cao hơn cả việc tối ưu thêm chunking.
