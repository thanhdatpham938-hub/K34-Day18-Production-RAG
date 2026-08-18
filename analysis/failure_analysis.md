# Failure Analysis — Lab 18: Production RAG

**Học viên:** Phạm Thanh Đạt (bài tập cá nhân)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8319 | 0.7225 | **-0.1094** |
| Answer Relevancy | 0.6854 | 0.6834 | -0.0020 |
| Context Precision | 0.9250 | 0.9500 | +0.0250 |
| Context Recall | 0.9083 | 0.7917 | **-0.1167** |

Chạy thật `python main.py` với OpenAI key hợp lệ (`sk-proj-...`), 20 câu hỏi từ `test_set.json`, model `gpt-4o-mini`.

**Kết quả bất ngờ:** Production pipeline (hierarchical chunking + enrichment + hybrid search + rerank) **thua** naive baseline (paragraph chunking + dense-only) ở faithfulness và context_recall, dù thắng ở context_precision. Chi tiết root cause ở phần Bottom-5 bên dưới — phần lớn không phải do M2/M3 (search/rerank) mà do M1 chunking (child 256 ký tự quá nhỏ, cắt đứt bảng/số liệu) và generation (gpt-4o-mini với prompt đơn giản `"Trả lời CHỈ dựa trên context"` không đủ mạnh để tổng hợp/tính toán từ nhiều chunk).

## Bottom-5 Failures

*(5 câu hỏi có avg score thấp nhất trong 4 metrics, lấy trực tiếp từ `reports/ragas_report.json` → `failures`, đã sort sẵn từ tệ nhất bởi `failure_analysis()`. Context bên dưới lấy từ chạy tay `HybridSearch`+`CrossEncoderReranker` trên đúng câu hỏi đó để xác minh nguyên nhân.)*

### #1
- **Question:** "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
- **Expected:** 15 ngày (v2024 hiện hành); v2023 = 12 ngày đã bị thay thế
- **Worst metric:** faithfulness — **0.0**
- **Context thật (rerank top-2):** `nghi_phep_nam_v2023.md` (score 0.994, "12 ngày phép năm") và `nghi_phep_nam_v2024.md` (score 0.993, "15 ngày phép năm") — **gần như đồng hạng**, cả 2 phiên bản mâu thuẫn cùng lọt vào context đưa cho LLM.
- **Error Tree:** Output sai → Context đúng (có v2024) nhưng **cũng có v2023 mâu thuẫn** → Root cause: retrieval không phân biệt được version hiện hành/đã bị thay thế, generation nhận context lẫn lộn 2 con số khác nhau cho cùng 1 câu hỏi → faithfulness checker không thể xác nhận claim nào được "support" rõ ràng vì context tự mâu thuẫn.
- **Root cause:** Không có metadata `version`/`superseded_by` được dùng để lọc/boost tại retrieval time.
- **Suggested fix:** Thêm field version vào metadata (M5 `extract_metadata` đã có cơ chế tương tự), loại bỏ hoặc downrank chunk có version cũ hơn khi phát hiện 2 chunks cùng chủ đề nhưng khác version. Test lại: sau fix, chỉ còn 1 trong 2 version lọt top-3, hoặc rerank score gap giữa 2 bản > 0.05 nghiêng về bản mới.

### #2
- **Question:** "Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"
- **Expected:** 5-50 triệu → Director phê duyệt; cần IT xác nhận cấu hình; cần ≥3 báo giá (vì >10tr)
- **Worst metric:** faithfulness — **0.0**
- **Context thật (rerank top-3):** top-1 (`mua_sam.md`, score 0.537) chỉ có 1 phần quy trình chung, cắt trước khi tới bảng ngưỡng phê duyệt cụ thể; **top-2 (`hoan_chi_dao_tao.md`, score 0.093) là chính sách hoàn chi đào tạo, KHÔNG liên quan** — nhưng lọt top-3 vì trùng con số "30.000.000 VNĐ" (tài trợ khóa học tối đa 30tr, không phải giá laptop).
- **Error Tree:** Output sai → Context sai/thiếu → Root cause: **keyword collision theo số liệu** — reranker bị con số "30 triệu" đánh lừa, kéo nhầm 1 document hoàn toàn khác chủ đề vào top-3, chiếm chỗ của chunk chứa bảng ngưỡng phê duyệt thật (5-50tr → Director).
- **Root cause:** Semantic/lexical match trên con số thuần túy không phân biệt được ngữ cảnh (giá laptop vs chi phí đào tạo); character-based child chunk cũng cắt mất bảng ngưỡng phê duyệt trước khi tới đoạn quan trọng.
- **Suggested fix:** Structure-aware chunking cho các bảng ngưỡng phê duyệt (giữ nguyên cả bảng, không cắt theo 256 ký tự); cân nhắc thêm metadata `category` (M5 đã extract field này) để loại trừ chunk khác category khi rerank.

### #3
- **Question:** "Nhân viên thử việc có được nghỉ phép năm không?"
- **Expected:** KHÔNG — nhân viên thử việc không được nghỉ phép năm
- **Worst metric:** faithfulness — **0.0**
- **Context thật (rerank top-1):** `thu_viec.md`, score **0.995** — chunk chứa đúng nguyên văn "...phép năm**. Trường hợp cần nghỉ việc riêng, nhân viên thử việc phải xin nghỉ không lương..." → **retrieval ĐÚNG, độ tin cậy rất cao.**
- **Error Tree:** Output sai → **Context đúng** (verify thủ công, retrieval không có lỗi) → Root cause nằm ở **generation**, không phải M1/M2/M3.
- **Root cause:** Đây là finding quan trọng nhất: context hoàn toàn chính xác nhưng faithfulness vẫn = 0. Nghi vấn: `gpt-4o-mini` với prompt đơn giản (`pipeline.py::run_query`, chỉ có 1 dòng system prompt "Trả lời CHỈ dựa trên context...") có thể đã diễn giải/thêm chi tiết không có trong context nguyên văn (vd. suy luận thêm lý do), khiến RAGAS faithfulness-checker (vốn so khớp từng claim với context) đánh giá là "unsupported".
- **Suggested fix:** Tighten prompt — yêu cầu rõ "chỉ paraphrase, không suy luận thêm"; giảm `temperature`; test lại bằng cách so sánh faithfulness của câu hỏi này riêng lẻ trước/sau khi sửa prompt (context giữ nguyên, chỉ đổi prompt).

### #4
- **Question:** "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
- **Expected:** Hạn 15 ngày, quá hạn 5 ngày → phí 2%/tháng trên 15.000.000 VNĐ, quy đổi ~50.000 VNĐ cho 5 ngày (pro-rata)
- **Worst metric:** faithfulness — **0.2**
- **Context thật (rerank top-2):** top-1 (`tam_ung.md`, score 0.983) có đúng "phí 2%/tháng"; top-2 (score 0.418) có đúng "hạn 15 ngày" — **cả 2 fact cần thiết đều có mặt trong top-3.**
- **Error Tree:** Output sai (một phần, score 0.2 not 0.0) → Context đúng đầy đủ → Root cause: generation phải **tự tính toán pro-rata** (2%/tháng → quy đổi cho 5 ngày) — phép tính không nằm sẵn trong context, đòi hỏi reasoning số học mà model nhỏ (gpt-4o-mini, prompt không yêu cầu show-work) dễ tính sai hoặc áp dụng nhầm công thức.
- **Root cause:** Generation-side numerical reasoning limitation, không phải retrieval.
- **Suggested fix:** Thêm few-shot example tính pro-rata trong system prompt, hoặc yêu cầu model liệt kê công thức trước khi ra số cuối (chain-of-thought ngắn) để tăng khả năng tính đúng và faithfulness (số liệu có thể trace ngược về context + phép tính tường minh).

### #5
- **Question:** "Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?"
- **Expected:** KHÔNG — phải báo cáo trong 1 giờ qua helpdesk/hotline CNTT
- **Worst metric:** answer_relevancy — **0.0**
- **Context thật (rerank top-1):** `bao_mat_su_co.md`, score **0.771** — chunk chứa đúng "Tuyệt đối **không tự ý xử lý malware**..." → **retrieval đúng.**
- **Error Tree:** Output sai → Context đúng → Root cause: answer_relevancy đo mức độ câu trả lời "khớp" với câu hỏi gốc (bằng cách sinh ngược câu hỏi từ answer rồi so embedding) — score 0.0 gợi ý answer có thể đã lan man/chung chung, hoặc rơi vào nhánh fallback "Không tìm thấy." của `pipeline.py` dù context có sẵn (do lỗi logic fallback hoặc do generation trả lời không trực diện vào câu hỏi "có nên tự xử lý không").
- **Suggested fix:** Kiểm tra lại answer thực tế được sinh ra cho câu hỏi này (hiện chưa log answer text ra file, chỉ log score) — thêm logging answer vào `save_report`/`EvalResult` để debug trực tiếp thay vì suy luận gián tiếp qua score.

## Case Study (cho presentation)

**Question chọn phân tích:** #3 — "Nhân viên thử việc có được nghỉ phép năm không?"

**Error Tree walkthrough:**
1. **Output đúng?** → Không (faithfulness = 0.0 theo RAGAS).
2. **Context đúng?** → **Có** — verify thủ công: chunk top-1 (score 0.995, `thu_viec.md`) chứa chính xác câu trả lời "KHÔNG được nghỉ phép năm".
3. **Query rewrite OK?** → Không liên quan — query đơn giản, rõ nghĩa, không cần rewrite.
4. **Fix ở bước:** **Generation** (LLM answer synthesis trong `pipeline.py::run_query`), không phải M1/M2/M3. Đây là bài học quan trọng nhất của lab: retrieval tốt (context_precision=0.95, cao hơn cả naive) không đảm bảo faithfulness cao (0.72, thấp hơn naive 0.83) — pipeline có nhiều bước hơn (enrichment, rerank) không tự động nghĩa là câu trả lời tốt hơn nếu prompt cuối cùng cho LLM không được tối ưu tương xứng.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Log full answer text (không chỉ score) vào `reports/ragas_report.json` để debug case #5 chính xác thay vì suy luận.
- Tighten generation prompt (case #1, #3, #4): thêm ràng buộc "chỉ paraphrase, nêu rõ nếu context mâu thuẫn giữa các version" + 1 ví dụ few-shot tính pro-rata.
- Table-aware chunking (case #2): không cắt bảng ngưỡng phê duyệt theo `child_size=256`.
