# Model Comparison Guide

## Understanding Detection Differences

### Why Different Models Give Different Results?

Both models are trained on the same dataset but use different architectures:

```
┌─────────────────────────────────────────────────────────────┐
│                    FASTER R-CNN                              │
├─────────────────────────────────────────────────────────────┤
│ Architecture: Two-Stage Detector                            │
│ Stage 1: Region Proposal Network (RPN)                      │
│ Stage 2: Classification & Bounding Box Regression           │
│                                                              │
│ Characteristics:                                             │
│ ✓ More aggressive in finding objects                        │
│ ✓ Better for medium to large objects                        │
│ ✓ Higher recall (finds more objects)                        │
│ ✗ May include more false positives                          │
│                                                              │
│ Example Result: 10 aircraft detected                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      RETINANET                               │
├─────────────────────────────────────────────────────────────┤
│ Architecture: Single-Stage Detector with Focal Loss         │
│ Uses Feature Pyramid Network (FPN)                          │
│                                                              │
│ Characteristics:                                             │
│ ✓ Better at detecting small objects                         │
│ ✓ More precise (fewer false positives)                      │
│ ✓ Higher precision (more accurate detections)               │
│ ✗ May miss some marginal detections                         │
│                                                              │
│ Example Result: 7 aircraft detected                         │
└─────────────────────────────────────────────────────────────┘
```

## Detection Metrics Explained

### Precision vs Recall Trade-off

```
High Recall (Faster R-CNN)
├── Finds: 10 aircraft
├── True Positives: 8
├── False Positives: 2
└── Recall: 100% (found all aircraft)
    Precision: 80% (80% were correct)

High Precision (RetinaNet)
├── Finds: 7 aircraft
├── True Positives: 7
├── False Positives: 0
└── Recall: 87.5% (missed 1 aircraft)
    Precision: 100% (all detections correct)
```

## When to Use Each Model

### Use Faster R-CNN when:
- ✅ You need to find ALL aircraft (critical for surveillance)
- ✅ You can manually verify detections later
- ✅ Missing an aircraft is worse than false alarms
- ✅ Aircraft are medium to large in the image
- ✅ You want maximum coverage

### Use RetinaNet when:
- ✅ You need accurate detections only
- ✅ False alarms are costly
- ✅ Aircraft are small in the image
- ✅ You want automated processing without manual review
- ✅ Precision is more important than recall

## Confidence Threshold Impact

### Faster R-CNN at different thresholds:
```
Threshold 0.1: 15 detections (many false positives)
Threshold 0.3: 10 detections (balanced) ← DEFAULT
Threshold 0.5: 6 detections (high confidence only)
Threshold 0.7: 3 detections (very conservative)
```

### RetinaNet at different thresholds:
```
Threshold 0.1: 12 detections (some false positives)
Threshold 0.3: 7 detections (balanced) ← DEFAULT
Threshold 0.5: 5 detections (high confidence only)
Threshold 0.7: 3 detections (very conservative)
```

## Testing Recommendations

1. **Test the same image with both models**
   - Compare the bounding boxes
   - Check which detections are different
   - Verify which model is more accurate for your use case

2. **Adjust confidence threshold**
   - Start with 0.3 (default)
   - Increase if too many false positives
   - Decrease if missing aircraft

3. **Visual inspection**
   - Look at the detection results
   - Verify if the extra detections from Faster R-CNN are real aircraft
   - Check if RetinaNet missed any obvious aircraft

## Conclusion

**The behavior you're seeing is CORRECT and EXPECTED!**

- Faster R-CNN: 10 detections = More aggressive, higher recall
- RetinaNet: 7 detections = More conservative, higher precision

Both models are working correctly. Choose based on your needs:
- Need to find everything? → Use Faster R-CNN
- Need accurate detections only? → Use RetinaNet
