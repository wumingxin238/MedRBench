#!/usr/bin/env python3
"""Convert embedded oracle_diagnosis_gemini.json to standard inference output format."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default='src/Inference/oracle_diagnosis_gemini.json',
        help='Embedded-format JSON (case_id -> fields + model key)',
    )
    parser.add_argument(
        '--output',
        default='src/Inference/oracle_diagnosis_gemini_standard.json',
        help='Standard format: case_id -> { model_name -> {content, input} }',
    )
    parser.add_argument('--model', default='gemini2-ft')
    args = parser.parse_args()

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    in_path = args.input if os.path.isabs(args.input) else os.path.join(project_root, args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)

    with open(in_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    standard = {}
    for case_id, case in raw.items():
        if args.model in case:
            standard[case_id] = {args.model: case[args.model]}

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(standard, f, ensure_ascii=False, indent=2)

    print(f'Wrote {len(standard)} cases to {out_path}')


if __name__ == '__main__':
    main()
