# Báo cáo Day 7

- Họ và tên: **Phạm Cát Tân**
- MSSV: **2A202603678**

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: dùng lại hai PDF Day 6: [GCE-MIL v1](https://arxiv.org/abs/2605.17456v1), giấy phép CC BY 4.0, và [Symb-xMIL v2](https://arxiv.org/abs/2606.06224v2), giấy phép CC BY-NC-ND 4.0 theo metadata PDF. Từ 57 trang vật lý, chọn 31 trang: GCE-MIL trang 1–9 và 15–28; Symb-xMIL trang 2–9. Dùng snapshot A–1.000 của Day 6: 135 chunk, tối đa 1.000 ký tự/chunk, overlap 50 ký tự, không ghép qua trang. Nguồn, giấy phép, checksum và phạm vi trang nằm trong [corpus.json](../../data/corpus.json); [pages.json](../../data/pages.json) và [chunks.json](../../data/chunks.json) giữ text, nguồn, trang và metadata.
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16.16, pgvector 0.8.7; exact cosine search trên bảng `day6_chunks`, strategy `A`, namespace `1a38c6b30f28f685e6bccebcc41b4db551e539805f67df1e07a78b6af274ff9f`. Dùng OpenRouter `qwen/qwen3-embedding-8b`, 4.096 chiều, giữ nguyên cấu hình Day 6. Query được thêm instruction `Given a question, retrieve relevant passages from scientific papers.`; document không thêm prefix. API không cung cấp revision trọng số cụ thể. Day 7 không tạo lại embedding tài liệu.
- Agent model, cách chạy và phân chia dev/test: OpenRouter `deepseek/deepseek-v4.1-flash-1017`, temperature 0, dùng native tool calling; môi trường Day 7 đã chạy là Python 3.15.2. Chạy từ thư mục gốc bằng `python -X utf8 starter/agentic.py ask "GCE-MIL đề xuất ba tiêu chí nào?"`; cài đặt chi tiết tại [hướng dẫn chạy](../../starter/README.md#bài-thực-hiện-của-phạm-cát-tân). [Benchmark](../../data/benchmark.json) có 24 câu tiếng Việt: dev gồm Q01–Q15 có đáp án và D16–D17 thiếu bằng chứng; test gồm T01–T05 có đáp án và T06–T07 thiếu bằng chứng. Dev có tái sử dụng câu Day 6; test là câu mới về các đáp án khác, cố định trước khi chạy. Hai split dùng cùng corpus, không cách ly theo paper.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: câu có đáp án có đáp án chuẩn và đoạn evidence trên đúng nguồn/trang; `gold` là các bằng chứng thay thế, có bổ sung vị trí trang 3 cho Q01 trước thực nghiệm Day 7. Câu thiếu bằng chứng có lý do và kiểm tra trên toàn bộ text 31 trang, không suy ra chỉ từ top-k. Agent chỉ nhận câu hỏi, filter theo phạm vi câu hỏi và kết quả công cụ, không nhận gold hoặc nhãn `answerable`. Hit@1/Hit@3 yêu cầu một chunk đúng nguồn/trang chứa trọn excerpt sau chuẩn hóa Unicode và khoảng trắng. Context Evidence Rate kiểm tra evidence trong context cuối, cho phép ghép phần chồng của các chunk cùng trang mà không tự điền khoảng trống. Các chỉ số này chỉ tính 15 câu dev hoặc 5 câu test có đáp án. Nhãn câu trả lời 0/1/2, citation và abstention được đọc đối chiếu có Codex hỗ trợ; chưa có người chấm độc lập. Năm trang evidence test đã được render từ PDF gốc và kiểm tra trực quan, ghi tại [source_audit.json](../../results/source_audit.json).

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (`baseline`) | Dense; top-3; budget 4.000 ký tự; tối đa 1 lượt tìm | Mốc so sánh bằng query gốc | dev | 5/15 (33,3%) | 10/15 (66,7%) | 10/15 (66,7%) | Trung vị 3,157 s/câu; 0,006229 USD |
| Phương pháp 2 (`lexical`) | TF-IDF cosine; top-3; budget 4.000; tối đa 1 lượt | Từ khóa có thể giữ được thuật ngữ mà dense bỏ sót | dev | 4/15 (26,7%) | 11/15 (73,3%) | 11/15 (73,3%) | 3,243 s/câu; 0,007330 USD |
| Phương pháp 3 (`hybrid`) | Dense + TF-IDF; RRF, hằng số 60; 20 candidate mỗi nhánh; top-3; budget 4.000; tối đa 1 lượt | Hợp nhất thứ hạng của hai phương pháp | dev | 5/15 (33,3%) | 10/15 (66,7%) | 10/15 (66,7%) | 3,116 s/câu; 0,006263 USD |
| Cải thiện 1 (`hybrid_context`) | Hybrid; 30 candidate mỗi nhánh; top-6; budget 7.500; tối đa 1 lượt | Lấy thêm ứng viên để context có đủ evidence | dev | 5/15 (33,3%) | 10/15 (66,7%) | 11/15 (73,3%) | 3,359 s/câu; 0,008748 USD |
| Cải thiện 2 (`hybrid_multiturn`) | Như cải thiện 1; tối đa 3 lượt; query tiếp theo do model chọn; ưu tiên lượt mới trong context | Tìm lại khi lượt đầu chưa đủ bằng chứng | dev | 5/15 (33,3%) | 10/15 (66,7%) | 13/15 (86,7%) | 4,499 s/câu; 0,010640 USD |
| Cấu hình đã chọn (`hybrid_multiturn`) | Giữ nguyên cấu hình đã chốt trên dev | Đánh giá sau khi chốt | test | 3/5 (60,0%) | 4/5 (80,0%) | 4/5 (80,0%) | 2,998 s/câu; 0,005204 USD |
| Baseline (`baseline`) | Dense; top-3; budget 4.000; tối đa 1 lượt | So sánh cùng bộ test | test | 3/5 (60,0%) | 4/5 (80,0%) | 4/5 (80,0%) | 4,211 s/câu; 0,003146 USD |

Mọi cấu hình dùng cùng corpus, snapshot và nhãn; không dùng reranker riêng. Filter `section_group` chỉ áp dụng khi câu hỏi nêu phạm vi nội dung chính hoặc phụ lục; cả ba phương pháp dùng cùng filter. RRF cộng theo thứ hạng, không cộng trực tiếp cosine dense với điểm TF-IDF. Budget tính bằng ký tự, gồm cả nhãn nguồn. Các cột Hit báo ranking của query gốc ở cùng cutoff 1/3, kể cả cấu hình lấy top-6. Query viết lại và ranking từng lượt nằm trong trace; Context Evidence Rate phản ánh context cuối của cả pipeline.

Chi phí trong bảng là tổng chat và query embedding do provider báo cho cả cấu hình/split, không phải chi phí mỗi câu; thời gian là trung vị tổng thời gian xử lý mỗi câu. Tổng 85 lần chạy agent trên dev là 332,51 s / 0,039209 USD; 14 lần chạy trên test là 61,96 s / 0,008350 USD. Tổng benchmark chính là 0,047559 USD, chưa gồm lượt thử bị dừng, chạy retrieval riêng và hỏi tương tác. Có dùng cache query Day 6 và cache provider; các số này không đại diện chạy lạnh. Không ghi nhận lỗi model hoặc database trong benchmark chính.

Phân tích lỗi dev:

- **Q08:** baseline lấy đoạn chung và mảnh cuối câu, thiếu cách kiểm tra faithfulness nên từ chối, điểm 0. Giả thuyết là từ khóa hoặc query tiếng Anh tìm đúng mục “Scope of theory”. TF-IDF và hybrid nhiều lượt lấy được đoạn trang 15 nêu keep/remove và CAMELYON-16 pixel annotations, đều đạt 2.
- **Q10:** baseline chỉ tìm thấy giới thiệu noisy-OR, chưa có cơ chế bão hòa/marginal gain, điểm 0. Tăng candidate/context và cho tìm lại lấy được giải thích ở trang 6 và 16; cả hai cấu hình cải thiện đạt 2. Điều này cho thấy lỗi xếp hạng và chọn context cũng quan trọng bên cạnh chunking.
- **Q06:** hybrid nhiều lượt tìm được ρ=0.05 và Table 16 trên BRACS, nhưng trả lời chưa nêu rõ validation sweep nên chỉ đạt 1. **Q01/Q07 của `hybrid_context`** có đủ context nhưng model ghép citation theo dạng `[ID; ID]` nên bị từ chối. **D16 của cấu hình này** nói không đủ bằng chứng nhưng thêm số end-to-end sai: 0,18× thay vì 0,92×; vì vậy abstention không được chấm đúng.

Tiêu chí chọn được ghi trong metadata lượt dev trước test: tối đa hóa `(tổng answer_grade/2 + số abstention đúng) / toàn bộ số câu`; nếu hòa, ưu tiên Context Evidence Rate cao hơn, ít lỗi hơn, thời gian trung vị thấp hơn rồi tên cấu hình. Điểm chọn lần lượt là baseline 13/17, lexical 15,5/17, hybrid 13,5/17, hybrid_context 13/17 và hybrid_multiturn 16,5/17. Chọn `hybrid_multiturn` lúc 16:31:02 ngày 14/11/2026 (UTC+7), lưu trong [selection.json](../../results/dev-final/selection.json), rồi mới chạy test. Không đổi sang baseline sau khi thấy baseline tốt hơn trên test.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: model gọi `search` để truy xuất bằng phương pháp đang thử và `submit_answer` để kết thúc hoặc từ chối. Lượt tìm đầu dùng nguyên câu hỏi; các lượt tiếp theo có thể viết lại bằng tiếng Anh từ thông tin đã tìm được. Đầu vào benchmark còn được tìm riêng bằng query gốc để đo ranking; lượt đo này không tính là tool call của model. Context loại trùng ID, giữ nguyên text/nguồn/trang, xen kẽ theo hạng giữa các lượt; cấu hình nhiều lượt ưu tiên lượt mới. Dừng khi model trả lời/từ chối hoặc hết lượt. Câu trả lời phải có `[chunk-id]` đầy đủ nằm trong context; thiếu/sai định dạng thì từ chối. Kiểm tra ID không tự chứng minh nguồn hỗ trợ nội dung. Nội dung tài liệu được coi là dữ liệu, không phải chỉ dẫn. CLI lưu trace và loại lỗi, không dùng câu trả lời hay vector giả khi backend lỗi.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| baseline / dev | 17/17 (100%) | 17 câu × 1 lượt | 4 / 0 / 11; 1,47/2 | 16/16 (100%) | 11/11 (100%) | 2/2 (100%) | 4/15 (26,7%) |
| lexical / dev | 17/17 (100%) | 17 câu × 1 lượt | 1 / 1 / 13; 1,80/2 | 23/24 (95,8%) | 14/14 (100%) | 2/2 (100%) | 1/15 (6,7%) |
| hybrid / dev | 17/17 (100%) | 17 câu × 1 lượt | 3 / 1 / 11; 1,53/2 | 16/17 (94,1%) | 12/12 (100%) | 2/2 (100%) | 3/15 (20,0%) |
| hybrid_context / dev | 17/17 (100%) | 17 câu × 1 lượt | 3 / 0 / 12; 1,60/2 | 24/24 (100%) | 12/12 (100%) | 1/2 (50,0%) | 3/15 (20,0%) |
| hybrid_multiturn / dev | 17/17 (100%) | 10 câu × 1; 4 câu × 2; 3 câu × 3 | 0 / 1 / 14; 1,93/2 | 33/35 (94,3%) | 15/15 (100%) | 2/2 (100%) | 0/15 (0%) |
| hybrid_multiturn / test | 7/7 (100%) | 5 câu × 1; 2 câu × 3 | 1 / 0 / 4; 1,60/2 | 4/4 (100%) | 4/4 (100%) | 2/2 (100%) | 1/5 (20,0%) |
| baseline / test | 7/7 (100%) | 7 câu × 1 lượt | 0 / 0 / 5; 2,00/2 | 6/6 (100%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0%) |

Grounded answer tính trên 15 câu dev hoặc 5 câu test có đáp án; câu bị từ chối chấm 0. Citation hợp lệ được tổng hợp từ danh sách `citations` của kết quả cuối và nhãn đọc đối chiếu; không đếm raw output đã bị chặn, cũng không bao quát mọi ID được model nhắc trong lời từ chối. “Câu có citation” có mẫu số là số kết quả `answered`. Vì vậy tỷ lệ citation 100% không có nghĩa mọi câu đã trả lời được: T05 của cấu hình được chọn bị chặn trước và được phản ánh ở điểm 0/từ chối nhầm.

Đã đọc đủ 5 câu test có đáp án cho cả baseline và cấu hình được chọn: context 0/1/2 lần lượt là 0/0/5 ở cả hai cấu hình. Context có thể đủ về nghĩa nhưng không chứa nguyên excerpt gold, nên đánh giá đọc đối chiếu 5/5 không đồng nhất với Context Evidence Rate tự động 4/5. Nhãn và lý do từng câu nằm trong [reviews test](../../results/test-final/reviews.json).

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| Q10 / baseline / dev | Có đáp án | 1 lượt; top-3 chỉ có đoạn giới thiệu, chưa có cơ chế marginal gain | Từ chối vì không đủ bằng chứng để giải thích | Answer 0; context 0. Sai ở truy xuất/context. |
| Q10 / hybrid_multiturn / dev | Có đáp án | 2 lượt; query tiếp theo `noisy-OR diminishing marginal returns redundant patches complementary evidence independence`; lấy được trang 6,16 | Noisy-OR bão hòa khi concept đã phủ; patch dư thừa có marginal gain thấp, nên ưu tiên bằng chứng bổ sung. Nguồn: `[d5b030ddecb30dd1068a838daccf18a5d4d708c033b7652bcf06f4ea97e95df7]`, `[4f7b3742b66151cfcf8b4ff68e77739fa6c79316220e5ff8bb0d9986e7a0533e]` | Answer 2; context 2. Câu trả lời đầy đủ và có nguồn hỗ trợ. |
| T01 / hybrid_multiturn / test | Có đáp án | 1 lượt; tìm được đoạn P1 trang 4 | Macro-F1 giảm 0.078, trung bình trên 9 dataset và 9 backbone `[18754f937c068ab1bcd8d8ce599d7f682f970f4a4ba8083bbbe0b98409112d36]` | Answer 2; context 2; citation hỗ trợ đủ số liệu và phạm vi. |
| T05 / hybrid_multiturn / test | Có đáp án, lỗi citation | 1 lượt; có đoạn mục 3.2 trang 4 trong context | Kết quả cuối: “Không đủ bằng chứng để trả lời.” Raw model viết `[85593f52]` thay vì ID đủ 64 ký tự nên bị chặn | Answer 0; context 2; từ chối nhầm. Baseline trả lời đúng câu này. |
| T06 / hybrid_multiturn / test | Thiếu bằng chứng | 3 lượt; không tìm được mã đăng ký ClinicalTrials.gov | Nêu không đủ bằng chứng, không tự tạo mã NCT; danh sách citation cuối rỗng | Abstention đúng; câu hỏi có kiểm tra sự thiếu thông tin trên corpus. |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: [dev](../../results/dev-final/) chứa 85 trace và nhãn của 5 cấu hình; [test](../../results/test-final/) chứa 14 trace và nhãn của baseline/cấu hình chọn. Mỗi file `baseline.jsonl`, `lexical.jsonl`, `hybrid.jsonl`, `hybrid_context.jsonl`, `hybrid_multiturn.jsonl` ứng với cấu hình đã chạy; test chỉ có hai cấu hình. Trace giữ query, filter, ranking, context, native tool call, model response, token/cost và thời gian. Bảng đầy đủ ở [experiments.csv](../../results/experiments.csv); tổng hợp tại [dev summary](../../results/dev-final/summary_reviewed.json) và [test summary](../../results/test-final/summary_reviewed.json). Các lệnh chạy và tổng hợp được điền trong [starter/README.md](../../starter/README.md#bài-thực-hiện-của-phạm-cát-tân).
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn: hybrid nhiều lượt tăng câu trả lời đủ trên dev từ 11/15 lên 14/15 và Context Evidence Rate từ 10/15 lên 13/15; trung vị thời gian tăng từ 3,157 lên 4,499 s/câu. Trên test mới, baseline đủ 5/5 còn cấu hình chọn đủ 4/5 do lỗi citation T05; cả hai từ chối đúng 2/2 câu thiếu evidence. Kết quả này chưa chứng minh hybrid tốt hơn ngoài dev. Mẫu test nhỏ, cùng hai paper với dev; gold chưa bao quát mọi bằng chứng tương đương; text PDF có thể mất bố cục/ngắt từ, chưa OCR/hiểu toàn bộ hình. Codex hỗ trợ triển khai, soạn benchmark và đọc nhãn; chưa có đánh giá độc lập bởi người chấm.

Sau benchmark, phần thực thi được chuyển vào các file starter có sẵn, giữ nguyên prompt/cấu hình và kết quả lịch sử. Kiểm tra hồi quy bằng 99 trace đã lưu cho cùng request, ranking PostgreSQL, context, answer và citation; 7 bảng tổng hợp khớp, 24 kiểm thử code qua. Đây là kiểm tra hồi quy, không phải lượt test độc lập mới. [refactor_verification.json](../../results/refactor_verification.json) ghi hash trước/sau và bản nhãn gốc trước khi sửa lỗi dấu tiếng Việt; điểm và nhãn boolean giữ nguyên. Selection cũ gắn hash code cũ nên không dùng trực tiếp để chạy benchmark mới sau khi sắp xếp code.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện. Bài nộp đánh giá phần cơ bản với 31 trang nội dung, không đề nghị 15 điểm bonus tài liệu từ 100 trang.
