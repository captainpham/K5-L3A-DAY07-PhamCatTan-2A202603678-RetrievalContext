# Starter Day 7

## Bài thực hiện của Phạm Cát Tân

**MSSV: 2A202603678.** Agent dùng OpenRouter `deepseek/deepseek-v4.1-flash-1017`; query embedding dùng `qwen/qwen3-embedding-8b`, 4.096 chiều, khớp index Day 6. Phần thực thi nằm trong `Controller.decide`, `Retriever.search` và `run_agent` của [agentic.py](agentic.py). [baseline.py](baseline.py) cung cấp TF-IDF; [agent_metrics.py](agent_metrics.py) tổng hợp retrieval/context và nhãn đã đọc. [src/run.py](../src/run.py) nối cấu hình, lệnh chạy và lưu kết quả.

### Môi trường và dữ liệu

Chạy các lệnh PowerShell từ thư mục gốc repo Day 7, không chạy trong `starter/`. Cần Python 3.11 trở lên, Docker và một OpenRouter API key có quyền dùng hai model trên. Khi tạo môi trường mới bằng `uv`:

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Nếu `.venv` đã có thì chỉ cần lệnh cài requirements. Các lệnh bên dưới gọi trực tiếp Python trong `.venv`, không bắt buộc activate.

Dùng lại [repo Day 6](https://github.com/captainpham/K5-L3A-DAY06-PhamCatTan-2A202603678-DataIngestion) đặt cạnh repo Day 7. `DAY6_DIR` mặc định là `../K5-L3A-DAY06-PhamCatTan-2A202603678-DataIngestion/starter`. Code đọc metadata index từ `results/20261113T124446335645Z/run.json` bên Day 6; cần giữ snapshot này. Các file `pages.jsonl`, `chunks_A.jsonl`, manifest, benchmark và PDF của Day 6 cần thiết nếu chạy lại bước chuẩn bị dữ liệu. [data/corpus.json](../data/corpus.json) ghi nguồn, checksum, giấy phép, phạm vi trang và namespace đã dùng; dữ liệu text và benchmark Day 7 đã có trong `data/`.

Database phải có snapshot A–1.000 gồm 135 chunk của namespace `1a38c6b30f28f685e6bccebcc41b4db551e539805f67df1e07a78b6af274ff9f`. Máy hiện tại kết nối `127.0.0.1:55433`, database `day6_lab`, user `day6`. Khởi động container đã có:

```powershell
docker start day6_student_starter-postgres-1
```

Nếu tạo container/volume mới, cần nạp lại dữ liệu bằng pipeline Day 6 trước; riêng các JSON text trong repo Day 7 không chứa vector để phục hồi database. CLI `inspect` sẽ kiểm tra database có khớp snapshot hay không. Dữ liệu này phụ thuộc đầu ra Day 6, không được tự thay bằng các file `example_*.json`.

### Cấu hình và hỏi agent

Tạo `.env` nếu chưa có, giữ file đang có nếu đã điền key:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Trong `.env`, `OPENROUTER_API_KEY` là key riêng của người chạy; `AGENT_MODEL` đã đặt đúng model nêu trên. `DATABASE_URL` để trống sẽ kế thừa cấu hình Day 6; có thể đặt DSN của database riêng khi cần. Key trống cũng kế thừa từ `.env` Day 6. Biến môi trường có giá trị được ưu tiên hơn file cấu hình. `.env` được Git bỏ qua; không ghi key vào báo cáo hoặc commit.

```powershell
.\.venv\Scripts\python.exe -X utf8 starter/agentic.py inspect
.\.venv\Scripts\python.exe -X utf8 starter/agentic.py ask "GCE-MIL đề xuất ba tiêu chí nào?"
```

`inspect` phải báo 135 chunk, 24 câu benchmark, embedding 4.096 chiều và đúng model agent. `ask` mặc định dùng `hybrid_multiturn`, in câu trả lời/status và đường dẫn trace `results/ask-*.json`. Muốn thử dense baseline:

```powershell
.\.venv\Scripts\python.exe -X utf8 starter/agentic.py ask "GCE-MIL đề xuất ba tiêu chí nào?" --config baseline
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Đã kiểm tra 24 test đều qua. Test fixture chỉ kiểm tra chức năng code, không thay thế benchmark model thật.

### Đọc và tổng hợp kết quả thực nghiệm

[experiments.csv](../results/experiments.csv) có đủ 5 cấu hình dev và 2 cấu hình test. [Báo cáo cá nhân](../report/individual/2A202603678_PhamCatTan.md) giải thích metric, nhãn và lỗi. Trace/nhãn đầy đủ ở [dev-final](../results/dev-final/) và [test-final](../results/test-final/). Tổng hợp lại các nhãn đã điền bằng code hiện tại:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.summarize_reviews --run results/dev-final --reviews results/dev-final/reviews.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.summarize_reviews --run results/test-final --reviews results/test-final/reviews.json
.\.venv\Scripts\python.exe -X utf8 -m scripts.export_results --runs results/dev-final results/test-final --output results/experiments.csv
```

Các lệnh này chỉ tính lại từ trace/nhãn sẵn có, không gọi model và không tự chấm nội dung. Nhãn được đọc đối chiếu có Codex hỗ trợ, chưa có người chấm độc lập.

### Chạy thực nghiệm mới

Cấu hình nằm trong [configs.json](../data/configs.json): `baseline`, `lexical`, `hybrid`, `hybrid_context`, `hybrid_multiturn`. Chạy dev vào thư mục mới:

```powershell
.\.venv\Scripts\python.exe -X utf8 starter/agentic.py benchmark --split dev --output results/dev-rerun
```

Lệnh này gọi API thật, lưu từng câu và không ghi đè kết quả chính đã lưu. Lượt benchmark chính trước khi sắp xếp code đã dùng các lệnh sau, chạy bằng Python trong môi trường ảo tại thời điểm đó:

```powershell
python -X utf8 -m src.run benchmark --split dev --output results/dev-final
python -X utf8 -m src.run select --run results/dev-final --reviews results/dev-final/reviews.json
python -X utf8 -m src.run benchmark --split test --selection results/dev-final/selection.json --output results/test-final
```

Đây là lệnh của lượt chạy lịch sử, không dùng để ghi đè dữ liệu hiện có. Sau đó phần chữ của nhãn được sửa lỗi dấu, còn điểm/nhãn boolean giữ nguyên; bản nhãn đúng thời điểm chốt được lưu tại `results/dev-final/reviews_original_encoding.json`. Hash code lịch sử và kiểm tra hồi quy 99 trace được ghi tại [refactor_verification.json](../results/refactor_verification.json).

Selection trong `dev-final` thuộc code trước khi sắp xếp vào starter nên code hiện tại sẽ từ chối dùng nó để chạy test mới. Khi làm thí nghiệm mới, cần nhãn đọc đối chiếu của chính lượt dev mới rồi chốt bằng lệnh `select`; không chép nhãn cũ sang câu trả lời mới. Test T01–T07 hiện đã được xem và phân tích, chỉ dùng để đối chiếu lịch sử; nếu dùng kết quả đó để cải thiện tiếp thì cần test mới cho đánh giá độc lập. Bài nộp giữ nguyên kết quả chính: baseline đủ 5/5 câu test có đáp án, cấu hình chọn trên dev đủ 4/5 do lỗi citation, cả hai từ chối đúng 2/2 câu thiếu evidence.

## Hướng dẫn gốc của starter

Phần dưới giữ hướng dẫn ban đầu của đề bài. Trạng thái triển khai, cấu hình và lệnh áp dụng cho bài của Phạm Cát Tân nằm ở phần trên.

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
