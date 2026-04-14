import { describe, it, expect } from 'vitest';
import { nms, type BoundingBox } from './nms.js';

describe('nms', () => {
  // T19: overlapping boxes → correct suppression (IoU threshold)
  it('T19: suppresses overlapping boxes above IoU threshold', () => {
    const boxes: BoundingBox[] = [
      { x: 0, y: 0, width: 100, height: 100, score: 0.9 }, // box 0 (highest score)
      { x: 10, y: 10, width: 100, height: 100, score: 0.8 }, // box 1 (overlaps box 0 heavily)
      { x: 200, y: 200, width: 100, height: 100, score: 0.7 }, // box 2 (no overlap)
    ];

    const kept = nms(boxes, 0.5);

    // Box 0 kept (highest score), box 1 suppressed (high IoU with box 0), box 2 kept (no overlap)
    expect(kept).toEqual([0, 2]);
  });

  it('T19: keeps all boxes when IoU below threshold', () => {
    const boxes: BoundingBox[] = [
      { x: 0, y: 0, width: 50, height: 50, score: 0.9 },
      { x: 100, y: 100, width: 50, height: 50, score: 0.8 },
      { x: 200, y: 200, width: 50, height: 50, score: 0.7 },
    ];

    const kept = nms(boxes, 0.5);
    expect(kept).toEqual([0, 1, 2]);
  });

  it('T19: suppresses all duplicates of same box', () => {
    const boxes: BoundingBox[] = [
      { x: 0, y: 0, width: 100, height: 100, score: 0.95 },
      { x: 0, y: 0, width: 100, height: 100, score: 0.9 },
      { x: 0, y: 0, width: 100, height: 100, score: 0.85 },
    ];

    const kept = nms(boxes, 0.5);
    expect(kept).toEqual([0]); // Only the highest-scoring duplicate survives
  });

  it('returns empty for empty input', () => {
    expect(nms([], 0.5)).toEqual([]);
  });

  it('returns single box unchanged', () => {
    const boxes: BoundingBox[] = [{ x: 10, y: 10, width: 50, height: 50, score: 0.99 }];
    expect(nms(boxes, 0.5)).toEqual([0]);
  });

  it('handles partial overlap correctly', () => {
    // Two boxes with exactly 50% overlap area
    const boxes: BoundingBox[] = [
      { x: 0, y: 0, width: 100, height: 100, score: 0.9 },   // area = 10000
      { x: 50, y: 0, width: 100, height: 100, score: 0.8 },   // area = 10000
      // Intersection: 50x100 = 5000, Union: 10000+10000-5000 = 15000
      // IoU = 5000/15000 = 0.333...
    ];

    // With threshold 0.3: should suppress (0.333 >= 0.3)
    expect(nms(boxes, 0.3)).toEqual([0]);

    // With threshold 0.4: should keep both (0.333 < 0.4)
    expect(nms(boxes, 0.4)).toEqual([0, 1]);
  });
});
