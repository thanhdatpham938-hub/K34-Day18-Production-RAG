# Individual Reflection — Lab 18

**Tên:** Phạm Thanh Đạt
**Module phụ trách:** M1 → M5 (bài tập cá nhân, làm đủ cả 5 module)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M1 (semantic/hierarchical/structure-aware chunking), M3 (CrossEncoder + FlashRank rerank), M4 (RAGAS eval wrapper + diagnostic-tree failure analysis), M5 (summarize/HyQA/contextual-prepend/metadata-extraction + combined single-call mode). M2 đã có sẵn implementation đầy đủ trong scaffold (không có TODO), verify lại bằng test.
- **Các hàm/class chính đã viết:** `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware` (`src/m1_chunking.py`); `CrossEncoderReranker._load_model/.rerank`, `FlashrankReranker.rerank` (`src/m3_rerank.py`); `evaluate_ragas`, `failure_analysis` (`src/m4_eval.py`); `summarize_chunk`, `generate_hypothesis_questions`, `contextual_prepend`, `extract_metadata`, `_enrich_single_call` (`src/m5_enrichment.py`). Ngoài ra fix 1 bug thật trong `main.py` (`os.rename` → `os.replace`, crash khi report file đích đã tồn tại).
- **Số tests pass:** 37/37 (`pytest tests/ -v` — test_m1: 13, test_m2: 5, test_m3: 5, test_m4: 4, test_m5: 10)

## 2. Mapping bài giảng → Code

| Lecture Concept | Module | Hàm cụ thể | Observation (số liệu thật) |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 trên 26 documents thật (`data/`) tạo ra **208 chunks** (avg 99 chars) — nhỏ hơn nhiều so với basic (**51 chunks**, avg 410 chars) vì threshold cao khiến câu chỉ gộp khi rất giống nhau về ngữ nghĩa. |
| Hierarchical parent-child | M1 | `chunk_hierarchical()` | 11 parents / 87 children từ cùng 26 documents — child avg 239 chars (đúng theo `HIERARCHICAL_CHILD_SIZE=256`), parent avg 1907 chars. Retrieve theo child nhỏ nhưng vẫn giữ `parent_id` để trace lại context lớn. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | Không tự viết (đã có sẵn) nhưng đã trace qua debugger — RRF giải quyết vấn đề BM25 và dense cho ra thang điểm không so sánh được (BM25 score không bound, cosine similarity 0-1) bằng cách chỉ dùng **rank**, không dùng raw score. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Benchmark thật trên CPU: **avg 506ms / query** (3 docs, `bge-reranker-v2-m3`, sau khi loại thời gian load model ra khỏi phép đo — lần đo đầu gồm cả load model ra tới 51.9s, phải tách warm-up riêng mới đo đúng). Với top-20 → top-3 mỗi query, latency này là chấp nhận được cho production nhưng đáng lo nếu scale lên top-50. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Sau khi có OpenAI key thật: Naive (faithfulness 0.83, answer_relevancy 0.69, context_precision 0.93, context_recall 0.91) vs Production (0.72 / 0.68 / 0.95 / 0.79). Trước đó với key sai (OpenRouter) đã phát hiện 1 điều thú vị: khi RAGAS fail ở **top-level** (key rỗng) code trả về 0.0, còn khi fail ở **từng job con** (key sai định dạng nhưng vẫn "hợp lệ cú pháp") thì trả về `NaN` — 2 chế độ fail khác nhau của cùng 1 nguyên nhân gốc. |
| Diagnostic tree / failure analysis | M4 | `failure_analysis()` | Bottom-5 thật (từ `reports/ragas_report.json`) cho thấy điều bất ngờ: **3/5 case context retrieval đúng 100% nhưng vẫn fail** (faithfulness hoặc answer_relevancy = 0.0) — chứng minh bằng cách chạy tay lại `HybridSearch`+`CrossEncoderReranker` trên đúng câu hỏi đó và thấy context top-1 chứa sẵn câu trả lời đúng. Root cause thật sự nằm ở generation prompt (`pipeline.py::run_query`), không phải M1-M3. |
| Contextual embeddings | M5 | `contextual_prepend()` | Fallback (`Trích từ {source}. {text}`) pass test `test_contextual_contains_original` — verify fallback path không làm mất nguyên văn gốc. Trong production run thật, `_enrich_single_call` (combined mode) chạy full 100/100 chunks thành công với key mới, ~4-5s/chunk (1 API call/chunk, gpt-4o-mini). |

## 3. Khó khăn & Cách giải quyết

Lỗi gặp phải đều nằm ở **môi trường**, không phải logic 5 module:

1. **`numpy` build từ source thất bại trên Python 3.13:** `langchain==0.2.17` pin `numpy>=1.26.0,<2.0.0` cho `python_version>=3.12`, nhưng numpy 1.26.4 không có wheel sẵn cho Python 3.13 → pip fallback build từ source → cần MSVC compiler không có sẵn trên máy. **Giải quyết:** cài Python 3.12 qua winget, tạo venv riêng.
2. **`WinError 32: process cannot access the file` giữa lúc `pip install`:** Venv ban đầu đặt trong thư mục OneDrive-synced → OneDrive khóa file giữa lúc pip đang ghi (~200 packages, bao gồm `torch` ~2.3GB). **Giải quyết:** xóa venv, tạo lại ở `C:\Users\<user>\venvs\...` ngoài OneDrive.
3. **`UnicodeEncodeError` khi print emoji (`📌`, `⚠️`) trên Windows console (`cp1258`):** Console Windows dùng codepage tiếng Việt không map được emoji. **Giải quyết:** set `PYTHONUTF8=1` và `PYTHONIOENCODING=utf-8` trước khi chạy script.
4. **`main.py` crash `os.rename` khi report file đích đã tồn tại (`WinError 183`):** Bug thật trong scaffold code — chạy `main.py` lần 2 thì `reports/ragas_report.json` đã tồn tại từ lần 1, `os.rename` trên Windows không tự overwrite. **Giải quyết:** đổi sang `os.replace` (overwrite atomic, đúng hành vi mong muốn).
5. **API key sai loại (OpenRouter thay vì OpenAI):** `.env` có key `sk-or-v1-...` — không tương thích với client `OpenAI()` mặc định (trỏ thẳng `api.openai.com`, không có `base_url`). Toàn bộ enrichment (M5) và RAGAS eval (M4) fail 401. **Đã giải quyết:** thay bằng OpenAI key thật (`sk-proj-...`), verify bằng 1 call test nhỏ trước khi chạy full pipeline (tránh lặp lại việc chạy 16 phút rồi mới biết key sai ở bước cuối). Chạy lại `python main.py` thành công, có RAGAS score thật.
- **Thời gian debug:** ~2 giờ cho 5 vấn đề trên (chủ yếu chờ pip install/model download qua mạng chập chờn + 2 lần chạy full pipeline ~16-18 phút/lần để xác nhận, không phải debug logic).

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Kiểm tra `.env` key hợp lệ (đúng provider) **trước** khi chạy bất kỳ pipeline nào tốn thời gian — tránh mất 16 phút chạy `main.py` chỉ để phát hiện key sai ở bước cuối cùng (RAGAS eval).
- **Module nào muốn thử tiếp:** Table-aware chunking cho M1 — 2/5 failure case trong `failure_analysis.md` (case #2, #4) đều do `chunk_hierarchical` cắt trúng bảng markdown/đoạn in đậm ngay tại số liệu cần trả lời. Muốn viết thêm logic detect block dạng `|...|` và không split nó theo `child_size`.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | — (bài cá nhân) |
| Problem solving | 5 (5 vấn đề môi trường liên tiếp, tự chẩn đoán root cause từng cái bằng log/network test thay vì đoán mò) |

## Action Plan cho project

### Hiện tại
- RAG pipeline hiện tại: Đã chạy full end-to-end với RAGAS score thật (naive vs production, xem `analysis/group_report.md`). Kết quả bất ngờ: production thua naive ở faithfulness (-0.109) và context_recall (-0.117), chỉ thắng context_precision (+0.025).
- Known issues (đã xác nhận bằng data thật, không phải giả định): (1) Chunking hiện tại (`chunk_hierarchical`) không table-aware, cắt trúng bảng/emphasis ở đúng điểm chứa câu trả lời (case #2 trong `failure_analysis.md`); (2) chưa có cơ chế xử lý version conflict giữa nhiều bản chính sách (case #1); (3) **quan trọng nhất** — generation prompt (`pipeline.py::run_query`) là bottleneck lớn hơn cả retrieval: 3/5 Bottom-5 fail dù context đúng 100% (case #3, #4, #5).

### Plan áp dụng
1. [ ] Chunking strategy: Hierarchical (parent 2048/child 256) làm mặc định, cộng thêm table-aware split (không cắt block markdown table) — vì đã thấy rõ hierarchical retrieve chính xác hơn basic (child nhỏ, tập trung 1 ý) nhưng vẫn thua khi gặp bảng.
2. [ ] Search: Hybrid BM25 (tiếng Việt qua `underthesea`) + Dense (`bge-m3`) qua RRF — vì BM25 bắt được exact-match từ khóa tiếng Việt (số ngày, tên chính sách) mà dense đôi khi bỏ lỡ, và ngược lại.
3. [ ] Reranking: Có — `bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder`, latency ~506ms/query (3 docs, CPU) chấp nhận được cho top-3.
4. [x] Evaluation: RAGAS với OpenAI key hợp lệ — **đã có số thật**, xem `analysis/group_report.md`. Phương pháp thủ công (chạy trực tiếp search+rerank, so context với ground truth) vẫn hữu ích để verify root cause của từng failure case (context đúng hay sai) mà bản thân RAGAS score không tự giải thích được.
5. [x] Enrichment: Combined single-call mode (M5) — đã chạy thật, 100/100 chunks thành công, 1 API call/chunk.
6. [ ] **Ưu tiên cao nhất tiếp theo — Generation prompt:** Thêm few-shot cho câu hỏi cần tính toán (pro-rata) và câu hỏi có version conflict; yêu cầu model paraphrase thay vì suy luận thêm. Đây là đòn bẩy lớn nhất còn lại vì 3/5 Bottom-5 fail dù context đúng.

### Timeline
- ~~Tuần 1 (25/08 - 31/08): Thay OpenAI key hợp lệ, chạy lại `python main.py` để có RAGAS score thật~~ — **Hoàn thành 19/08**, sớm hơn dự kiến. So với chẩn đoán thủ công ban đầu: 8/10 failure category khớp (đúng hướng context_recall/precision), nhưng finding lớn nhất (3/5 case fail dù context đúng) chỉ RAGAS mới phát hiện được — chẩn đoán thủ công không đo được faithfulness/answer_relevancy vì không có real LLM answer để so sánh.
- Tuần 1 (25/08 - 31/08): Tighten generation prompt theo case #1/#3/#4, chạy lại RAGAS, so faithfulness trước/sau.
- Tuần 2 (01/09 - 07/09): Viết table-aware chunking cho case #2, đo lại rerank score (kỳ vọng > 0.8 sau fix).
