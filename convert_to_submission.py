#!/usr/bin/env python3
"""
將 test_pred.json 的 video_id 轉換為比賽評分器要求的格式，輸出 test_trans_pred.json
無需重新跑推論。
"""
import argparse
import json
from pathlib import Path

VIDEO_ID_MAP = {
    "ukdd_navi_00051": "vid_001",
    "ukdd_navi_00068": "vid_002",
    "ukdd_navi_00076": "vid_003",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_pred.json")
    parser.add_argument("--output", default="test_trans_pred.json")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        pred = json.load(f)

    for v in pred["videos"]:
        old_id = v["video_id"]
        v["video_id"] = VIDEO_ID_MAP.get(old_id, old_id)

    with open(args.output, "w") as f:
        json.dump(pred, f, indent=2)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
