"""Tổng hợp nhãn đã điền cho một lượt chạy; không tự chấm câu trả lời bằng ID."""
import argparse

from starter.agent_metrics import summarize
from src.prepare import read_lines, write_json
from src.run import validate_cases
from src.settings import ROOT, read_json


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reviews', required=True, type=Path)
    args = parser.parse_args()
    meta = read_json(args.run / 'run.json')
    cases = validate_cases(read_json(ROOT / 'data/benchmark.json'), meta['split'])
    reviews = read_json(args.reviews)
    summaries = {}
    for name in meta['configs']:
        traces = read_lines(args.run / f'{name}.jsonl')
        if len(traces) != len(cases) or {t['query_id'] for t in traces} != {q['id'] for q in cases}:
            raise ValueError(f'{name}: thiếu hoặc trùng trace.')
        labels = [r for r in reviews if r['config'] == name]
        summaries[name] = summarize(cases, traces, labels)
    write_json(args.run / 'summary_reviewed.json', summaries)
    print('Saved:', args.run / 'summary_reviewed.json')


if __name__ == '__main__':
    main()
