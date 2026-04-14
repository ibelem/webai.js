/**
 * Non-Maximum Suppression (NMS) for object detection bounding boxes.
 */

export interface BoundingBox {
  /** x coordinate of top-left corner */
  x: number;
  /** y coordinate of top-left corner */
  y: number;
  /** box width */
  width: number;
  /** box height */
  height: number;
  /** confidence score */
  score: number;
}

/**
 * Compute Intersection over Union (IoU) between two boxes.
 */
function iou(a: BoundingBox, b: BoundingBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = a.width * a.height;
  const areaB = b.width * b.height;
  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}

/**
 * Filter overlapping bounding boxes using Non-Maximum Suppression.
 *
 * @param boxes - Array of bounding boxes with scores
 * @param iouThreshold - IoU threshold above which overlapping boxes are suppressed (0-1)
 * @returns Indices of boxes that survived suppression, sorted by score descending
 */
export function nms(boxes: BoundingBox[], iouThreshold: number): number[] {
  if (boxes.length === 0) return [];

  // Sort by score descending
  const indices = Array.from({ length: boxes.length }, (_, i) => i);
  indices.sort((a, b) => boxes[b].score - boxes[a].score);

  const kept: number[] = [];
  const suppressed = new Set<number>();

  for (const idx of indices) {
    if (suppressed.has(idx)) continue;
    kept.push(idx);

    for (const other of indices) {
      if (other === idx || suppressed.has(other)) continue;
      if (iou(boxes[idx], boxes[other]) >= iouThreshold) {
        suppressed.add(other);
      }
    }
  }

  return kept;
}
