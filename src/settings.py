"""Đọc cấu hình local và xác minh snapshot Day 6 trước khi truy xuất."""
import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    # Hash nội dung để selection không bị dùng lẫn giữa các lần đổi dữ liệu/code.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_settings():
    local = dotenv_values(ROOT / '.env')
    day6 = (ROOT / (os.getenv('DAY6_DIR') or local.get('DAY6_DIR') or
                    '../K5-L3A-DAY06-PhamCatTan-2A202603678-DataIngestion/starter')).resolve()
    inherited = dotenv_values(day6 / '.env')
    # Chỉ giá trị đã điền mới ghi đè; không ghi key ra log hoặc artifact.
    config = {**inherited, **{k: v for k, v in local.items() if v},
              **{k: v for k, v in os.environ.items() if v}}
    run_path = day6 / 'results/20261113T124446335645Z/run.json'
    run = read_json(run_path)
    if run['config']['strategies']['A'] != {'method': 'fixed', 'size': 1000, 'overlap': 50, 'filter': False}:
        raise ValueError('Snapshot Day 6 không phải A–1000 đã chọn.')
    return {
        'day6': day6, 'run': run, 'database_url': config.get('DATABASE_URL', ''),
        'api_key': config.get('OPENROUTER_API_KEY', ''),
        'agent_model': config.get('AGENT_MODEL', 'deepseek/deepseek-v4.1-flash-1017'),
        'namespace': run['experiment_fingerprint'], 'strategy': 'A',
        'dimensions': run['dimensions'], 'embedding': run['embedding'],
    }
