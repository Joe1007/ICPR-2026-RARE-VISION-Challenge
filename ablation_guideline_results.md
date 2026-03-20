| Exp ID | Variant | Backbone(s) | Fusion | Calibration | Temporal Decoding | Event Gen | mAP@0.5 | mAP@0.95 |
|-------|---------|-------------|--------|-------------|-------------------|-----------|---------|----------|
| A0 | Full | EndoFM+DINOv3 | weighted | T+TT | full | per-label | 0.4730 | 0.3658 |
| A2 | DINOv3 only | DINOv3 | single-backbone | T+TT | full | per-label | 0.4183 | 0.3309 |
| A3 | EndoFM only | EndoFM | single-backbone | T+TT | full | per-label | 0.3503 | 0.2845 |
| A4 | Uniform backbone weights | EndoFM+DINOv3 | uniform | T+TT | full | per-label | 0.4520 | 0.3492 |
| B1 | Uniform model weights | EndoFM+DINOv3 | weighted | T+TT | full | per-label | 0.4799 | 0.3388 |
| C1 | No struct. decoding | EndoFM+DINOv3 | weighted | T+TT | none | per-label | 0.3002 | 0.2612 |
| C8 | Tuple event gen | EndoFM+DINOv3 | weighted | T+TT | full | tuple-based | 0.3897 | 0.3333 |
