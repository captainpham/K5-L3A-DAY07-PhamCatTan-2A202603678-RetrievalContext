"""Xuất bảng CSV từ summary đã review, không tạo hoặc thay đổi số liệu đánh giá."""
import argparse
import csv
from pathlib import Path

from src.settings import read_json


def fraction(metric):
    return f"{metric['n']}/{metric['N']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for directory in args.runs:
        meta = read_json(directory / 'run.json')
        summaries = read_json(directory / 'summary_reviewed.json')
        for name, summary in summaries.items():
            config = meta['configurations'][name]
            retrieval, agent = summary['retrieval'], summary['agent']
            rows.append({
                'config': name, 'split': meta['split'], 'method': config['method'],
                'top_k': config['top_k'], 'budget_chars': config['budget'],
                'max_searches': config['max_searches'],
                'hit_at_1': fraction(retrieval['hit1']), 'hit_at_3': fraction(retrieval['hit3']),
                'context_evidence': fraction(retrieval['context_evidence']),
                'answer_grade_0': agent['grounded_answer']['0'],
                'answer_grade_1': agent['grounded_answer']['1'],
                'answer_grade_2': agent['grounded_answer']['2'],
                'mean_answer_grade_out_of_2': agent['grounded_answer']['mean'],
                'valid_citations': fraction(agent['citation_validity']),
                'tool_use': fraction(agent['tool_use']),
                'correct_abstention': fraction(agent['abstention']),
                'false_abstention': fraction(agent['false_abstention']),
                'errors': summary['errors'], 'median_ms': summary['median_ms'],
                'chat_cost_usd_reported': summary['chat_cost_usd_reported'],
                'embedding_cost_usd_reported': summary['embedding_usage']['reported_cost_usd'],
                'chat_calls_without_cost': summary['chat_calls_without_cost'],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
