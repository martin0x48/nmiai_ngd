"""Co-occurrence rescorer for shelf product classification.

Rescores low-confidence classifier predictions using spatial co-occurrence
context from nearby high-confidence detections. Never adds or removes
detections — only potentially changes category_id and score.

Usage:
    from cooccurrence_rescorer import CooccurrenceRescorer

    rescorer = CooccurrenceRescorer("data/cooccurrence.json")
    rescored = rescorer.rescore_image_predictions(detections)
"""

import json
from math import sqrt

import numpy as np


class CooccurrenceRescorer:
    """Rescore classification predictions using shelf co-occurrence context.

    Each detection must have:
        bbox: [x, y, w, h]  (COCO format)
        category_id: int
        det_score: float     (detector confidence)
        cls_score: float     (classifier top-1 probability)
        cls_probs: dict[int, float]  (top-k {category_id: probability})
    """

    def __init__(
        self,
        cooccurrence_path: str,
        alpha: float = 0.7,
        conf_threshold: float = 0.5,
        k_neighbors: int = 8,
        spatial_weight: float = 0.3,
        top_k_candidates: int = 5,
    ):
        self.alpha = alpha
        self.conf_threshold = conf_threshold
        self.k_neighbors = k_neighbors
        self.spatial_weight = spatial_weight
        self.top_k_candidates = top_k_candidates

        with open(cooccurrence_path) as f:
            data = json.load(f)

        self.image_cooc = np.array(data["image_cooccurrence"])
        self.spatial_cooc = np.array(data["spatial_cooccurrence"])
        self.num_categories = self.image_cooc.shape[0]

        # Precompute combined matrix
        self.combined = (
            (1 - spatial_weight) * self.image_cooc
            + spatial_weight * self.spatial_cooc
        )

    @staticmethod
    def _bbox_center(bbox):
        """Return (cx, cy) from COCO [x, y, w, h]."""
        x, y, w, h = bbox
        return x + w / 2, y + h / 2

    @staticmethod
    def _distance(c1, c2):
        return sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _context_score(self, neighbor_cat_ids, candidate_cat_id):
        """Mean co-occurrence probability of candidate given neighbors."""
        if not neighbor_cat_ids:
            return 0.0
        if candidate_cat_id >= self.num_categories:
            return 0.0
        scores = []
        for nid in neighbor_cat_ids:
            if nid < self.num_categories:
                scores.append(self.combined[nid][candidate_cat_id])
        return float(np.mean(scores)) if scores else 0.0

    def rescore_image_predictions(self, detections):
        """Rescore detections for a single image.

        Returns a new list of detections with potentially updated category_id
        and score. Never adds or removes detections.
        """
        if not detections:
            return []

        # Separate anchors (high confidence) from candidates (low confidence)
        anchors = []
        candidates = []
        for i, det in enumerate(detections):
            if det["cls_score"] >= self.conf_threshold:
                anchors.append((i, det))
            else:
                candidates.append((i, det))

        # If no anchors, return unchanged
        if not anchors:
            return [self._to_output(d) for d in detections]

        # If no candidates to rescore, return unchanged
        if not candidates:
            return [self._to_output(d) for d in detections]

        anchor_centers = [
            (self._bbox_center(d["bbox"]), d["category_id"]) for _, d in anchors
        ]

        result = list(detections)  # shallow copy

        for idx, det in candidates:
            det_center = self._bbox_center(det["bbox"])

            # Find k nearest anchors
            dists = [
                (self._distance(det_center, ac), cat_id)
                for ac, cat_id in anchor_centers
            ]
            dists.sort(key=lambda x: x[0])
            neighbor_cats = [cat_id for _, cat_id in dists[: self.k_neighbors]]

            # Get candidate categories from classifier top-k probs
            cls_probs = det.get("cls_probs", {})
            if not cls_probs:
                continue

            # Score each candidate
            best_cat = det["category_id"]
            best_score = -1.0

            for cat_id, cls_prob in cls_probs.items():
                ctx = self._context_score(neighbor_cats, cat_id)
                blended = self.alpha * cls_prob + (1 - self.alpha) * ctx
                if blended > best_score:
                    best_score = blended
                    best_cat = cat_id

            # Update detection
            new_det = dict(det)
            new_det["category_id"] = best_cat
            # Use the classifier probability for the chosen category
            chosen_cls_score = cls_probs.get(best_cat, det["cls_score"])
            new_det["score"] = round(det["det_score"] * chosen_cls_score, 4)
            new_det["rescored"] = best_cat != det["category_id"]
            result[idx] = new_det

        return [self._to_output(d) for d in result]

    @staticmethod
    def _to_output(det):
        """Convert to standard prediction format."""
        return {
            "image_id": det["image_id"],
            "category_id": det["category_id"],
            "bbox": det["bbox"],
            "score": det.get("score", round(det["det_score"] * det["cls_score"], 4)),
        }

    def rescore_all(self, predictions_by_image):
        """Rescore all predictions grouped by image_id.

        Args:
            predictions_by_image: dict[int, list[dict]] — image_id -> detections

        Returns:
            Flat list of rescored predictions in standard format.
        """
        all_preds = []
        stats = {"total": 0, "rescored": 0, "changed": 0}

        for img_id, dets in predictions_by_image.items():
            rescored = self.rescore_image_predictions(dets)
            all_preds.extend(rescored)
            stats["total"] += len(dets)

            # Count changes
            for orig, new in zip(dets, rescored):
                if orig.get("cls_score", 1.0) < self.conf_threshold:
                    stats["rescored"] += 1
                    if orig["category_id"] != new["category_id"]:
                        stats["changed"] += 1

        return all_preds, stats
