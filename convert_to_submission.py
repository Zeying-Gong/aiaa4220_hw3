#!/usr/bin/env python3
"""
Convert MMDetection test.bbox.json to Kaggle submission format
Usage:
   # Recommended: specify test.json to ensure all images are included
   python convert_to_submission.py --pred mm/work_dirs/deformable-detr_r50/test.bbox.json --output submission.csv

   # Or explicitly specify test.json path
   python convert_to_submission.py --pred mm/work_dirs/deformable-detr_r50/test.bbox.json --test data/test.json --output submission.csv

   # Or with image list
   python convert_to_submission.py --pred mm/work_dirs/deformable-detr_r50/test.bbox.json --images test_images.txt --output submission.csv

   # Auto-detect (NOT recommended - may miss images with no predictions)
   python convert_to_submission.py --pred mm/work_dirs/deformable-detr_r50/test.bbox.json --no-test --output submission.csv
Input format (test.bbox.json):
   [
       {"image_id": 123, "category_id": 0, "bbox": [x, y, w, h], "score": 0.95},
       ...
   ]
Output format (submission.csv):
   id,predictions
   123,"[{\"image_id\": 123, \"category_id\": 0, \"bbox\": [x, y, w, h], \"score\": 0.95}, ...]"
   456,"[]"  # Empty predictions for images with no detections
   ...
"""
import json
import argparse
import pandas as pd
from collections import defaultdict
import os
def convert_to_submission(pred_file, output_file, image_ids=None, test_json=None):
   """Convert predictions to Kaggle submission format"""

   # Load predictions
   with open(pred_file, 'r') as f:
       predictions = json.load(f)

   print(f"Loaded {len(predictions)} predictions from {pred_file}")

   # Group predictions by image_id
   pred_by_img = defaultdict(list)
   for pred in predictions:
       pred_by_img[pred['image_id']].append(pred)

   # Get unique image IDs
   if image_ids is None:
       if test_json:
           # Load from test.json to ensure ALL images are included
           with open(test_json, 'r') as f:
               test_data = json.load(f)
           image_ids = [img['id'] for img in test_data['images']]
           print(f"Loaded {len(image_ids)} image IDs from {test_json}")
       else:
           # Fallback: only use images with predictions (may miss some!)
           image_ids = sorted(pred_by_img.keys())
           print(f"WARNING: Using only images with predictions. This may miss images!")

   print(f"Processing {len(image_ids)} unique images")

   # Create submission rows
   rows = []
   images_with_preds = 0
   images_without_preds = 0

   for img_id in image_ids:
       img_preds = pred_by_img.get(img_id, [])
       if img_preds:
           images_with_preds += 1
       else:
           images_without_preds += 1
       rows.append({
           'id': img_id,
           'predictions': json.dumps(img_preds)
       })

   # Save as CSV
   submission_df = pd.DataFrame(rows)
   submission_df.to_csv(output_file, index=False)

   print(f"\n✓ Saved submission to {output_file}")
   print(f"  - {len(rows)} rows (one per image)")
   print(f"  - {images_with_preds} images with predictions")
   print(f"  - {images_without_preds} images with empty predictions")
   print(f"  - {sum(len(pred_by_img[img_id]) for img_id in image_ids)} total predictions")

   return submission_df
def load_image_ids(image_list_file):
   """Load image IDs from a text file (one ID per line)"""
   with open(image_list_file, 'r') as f:
       return [int(line.strip()) for line in f if line.strip()]
def main():
   parser = argparse.ArgumentParser(
       description='Convert MMDetection predictions to Kaggle submission format',
       formatter_class=argparse.RawDescriptionHelpFormatter,
       epilog=__doc__
   )
   parser.add_argument('--pred', required=True, help='Path to test.bbox.json')
   parser.add_argument('--output', default='submission.csv', help='Output CSV file')
   parser.add_argument('--test', default='data/test.json', help='Path to test.json (default: data/test.json)')
   parser.add_argument('--no-test', action='store_true', help='Do not use test.json (may miss images without predictions)')
   parser.add_argument('--images', help='Optional: text file with image IDs (one per line)')

   args = parser.parse_args()

   # Determine test.json path
   test_json = None
   if not args.no_test:
       if args.images:
           # If images list is provided, use it instead of test.json
           test_json = None
       elif os.path.exists(args.test):
           test_json = args.test
       else:
           print(f"WARNING: test.json not found at {args.test}")
           print("Will only use images with predictions. Use --no-test to suppress this warning.")
           test_json = None

   # Load image IDs if provided
   image_ids = None
   if args.images:
       image_ids = load_image_ids(args.images)
       print(f"Loaded {len(image_ids)} image IDs from {args.images}")

   # Convert
   convert_to_submission(args.pred, args.output, image_ids, test_json)

   print("\n✓ Done! You can now submit", args.output, "to Kaggle")
if __name__ == '__main__':
   main()
