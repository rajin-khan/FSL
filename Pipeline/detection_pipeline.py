import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import time
from ultralytics import YOLO
from collections import Counter
import sys

class YOLOv8DetectionPipeline:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45, img_size=640):
        """
        Initialize the YOLOv8 detection pipeline.
        
        Args:
            model_path (str): Path to the fine-tuned YOLOv8 model weights
            conf_thresh (float): Confidence threshold for detections
            iou_thresh (float): IoU threshold for non-maximum suppression
            img_size (int): Image size for detection
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        
        # Load model
        print(f"Loading YOLOv8 model from {model_path}")
        self.model = YOLO(model_path)
        
        # Extract class names directly from the model
        self.class_names = self.model.names
        
        # If class names weren't loaded from the model, try to load from external files
        if not self.class_names:
            print("Class names not found in model. Trying to load from external files...")
            self.class_names = self._get_class_names(model_path)
            
        print(f"Loaded {len(self.class_names)} class names: {self.class_names}")
        
    def _get_class_names(self, model_path):
        """Try to extract class names from the model directory."""
        # First, try to find a data.yaml file in the parent directory
        model_dir = Path(model_path).parent
        yaml_path = model_dir / 'data.yaml'
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    return data['names']
        
        # If no YAML file found, try to look for classes.txt
        txt_path = model_dir / 'classes.txt'
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # If no class names found, return None
        print("Warning: Could not find class names. Objects will be labeled with their class index.")
        return {i: f"Class {i}" for i in range(100)}  # Default fallback names
    
    def process_image(self, image_path, output_path=None, visualize=True):
        """
        Process a single image with the YOLOv8 model.
        
        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save the output image
            visualize (bool): Whether to visualize detections
            
        Returns:
            list: List of detections, each as [class_id, confidence, x, y, width, height]
            dict: Count of objects per class
            float: Average confidence of detections
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return [], {}, 0.0
        
        # Perform detection
        results = self.model(img, conf=self.conf_thresh, iou=self.iou_thresh, imgsz=self.img_size)
        
        # Process results
        detections = []
        class_counts = Counter()
        total_confidence = 0.0
        
        # Create a copy of the image for visualization
        img_vis = img.copy() if visualize else None
        
        # Parse results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class ID and confidence
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = self.class_names[cls_id]
                
                # Update class count
                class_counts[class_name] += 1
                
                # Add to total confidence
                total_confidence += conf
                
                # Convert to [class_name, confidence, x, y, width, height] format
                width = x2 - x1
                height = y2 - y1
                
                detections.append([class_name, conf, x1, y1, width, height])
                
                # Draw bounding box if visualize is True
                if visualize:
                    label = f"{class_name} {conf:.2f}"
                    
                    # Draw bounding box and label
                    color = (0, 255, 0)  # Green color for bounding box
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img_vis, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(detections) if detections else 0.0
        
        # Add class counts and average confidence to the image
        if visualize and img_vis is not None:
            y_offset = 30
            # Add detection quality info
            quality_text = f"Detection Quality: {self.get_accuracy_label(avg_confidence)}"
            quality_percentage = f"Est. Accuracy: {self.get_accuracy_percentage(avg_confidence):.1f}%"
            cv2.putText(img_vis, quality_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            cv2.putText(img_vis, quality_percentage, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            
            # Add class counts
            if class_counts:
                for class_name, count in class_counts.items():
                    text = f"{class_name}: {count}"
                    cv2.putText(img_vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30
        
        # Save the output image if output_path is provided
        if output_path and visualize:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_vis)
        
        return detections, dict(class_counts), avg_confidence
    
    def get_accuracy_percentage(self, avg_confidence):
        """
        Convert average confidence to an accuracy percentage.
        
        Args:
            avg_confidence (float): Average confidence of detections
            
        Returns:
            float: Estimated accuracy percentage
        """
        # Scale confidence to a more intuitive range (typically confidence >0.8 is very good)
        if avg_confidence > 0.8:
            # Scale 0.8-1.0 to 90-100%
            return 90 + (avg_confidence - 0.8) * 50
        elif avg_confidence > 0.6:
            # Scale 0.6-0.8 to 80-90%
            return 80 + (avg_confidence - 0.6) * 50
        elif avg_confidence > 0.4:
            # Scale 0.4-0.6 to 70-80%
            return 70 + (avg_confidence - 0.4) * 50
        else:
            # Scale 0.25-0.4 to 50-70%
            return max(50, 50 + (avg_confidence - 0.25) * 133)
    
    def get_accuracy_label(self, avg_confidence):
        """
        Convert average confidence to a human-readable label.
        
        Args:
            avg_confidence (float): Average confidence of detections
            
        Returns:
            str: Human-readable label for the quality of detections
        """
        if avg_confidence > 0.8:
            return "Excellent"
        elif avg_confidence > 0.6:
            return "Good"
        elif avg_confidence > 0.4:
            return "Moderate"
        else:
            return "Low"
    
    def process_directory(self, input_dir, output_dir=None, visualize=True, save_results=True):
        """
        Process all images in a directory.
        
        Args:
            input_dir (str): Path to the input directory containing images
            output_dir (str, optional): Path to save output images
            visualize (bool): Whether to visualize detections
            save_results (bool): Whether to save detection results to a text file
            
        Returns:
            dict: Dictionary mapping image paths to lists of detections
            dict: Dictionary mapping image paths to class counts
            dict: Dictionary mapping image paths to average confidence scores
        """
        # Create output directory if not exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files in the input directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(input_dir).glob(f'**/*{ext}')))
        
        # Process each image
        results = {}
        all_class_counts = {}
        all_confidences = {}
        total_counts = Counter()
        
        for img_path in tqdm(image_files, desc="Processing images"):
            # Determine output path
            rel_path = img_path.relative_to(input_dir)
            if output_dir:
                out_img_path = Path(output_dir) / rel_path
                os.makedirs(out_img_path.parent, exist_ok=True)
            else:
                out_img_path = None
            
            # Process the image
            detections, class_counts, avg_confidence = self.process_image(
                str(img_path),
                str(out_img_path) if out_img_path else None,
                visualize
            )
            
            # Store results
            results[str(img_path)] = detections
            all_class_counts[str(img_path)] = class_counts
            all_confidences[str(img_path)] = avg_confidence
            
            # Update total counts
            total_counts.update(class_counts)
            
            # Save results to a text file if requested
            if save_results and output_dir:
                txt_path = out_img_path.with_suffix('.txt')
                with open(txt_path, 'w') as f:
                    f.write(f"Objects detected in {img_path.name}:\n")
                    for class_name, count in class_counts.items():
                        f.write(f"{class_name}: {count}\n")
                    f.write(f"\nEstimated Detection Accuracy: {self.get_accuracy_percentage(avg_confidence):.1f}%\n")
                    f.write(f"Quality: {self.get_accuracy_label(avg_confidence)}\n")
                    f.write("\nDetailed detections:\n")
                    for det in detections:
                        # Format: class_name confidence x y width height
                        f.write(f"{det[0]} {det[1]:.6f} {det[2]} {det[3]} {det[4]} {det[5]}\n")
        
        # Calculate overall average confidence
        overall_avg_confidence = sum(all_confidences.values()) / len(all_confidences) if all_confidences else 0.0
        
        # Create a summary of all class counts and accuracy
        if output_dir and save_results:
            with open(os.path.join(output_dir, "detection_summary.txt"), 'w') as f:
                f.write("===== DETECTION SUMMARY =====\n\n")
                f.write(f"Overall Estimated Accuracy: {self.get_accuracy_percentage(overall_avg_confidence):.1f}%\n")
                f.write(f"Detection Quality: {self.get_accuracy_label(overall_avg_confidence)}\n")
                f.write("\n===== TOTAL OBJECT COUNTS ACROSS ALL IMAGES =====\n")
                for class_name, count in total_counts.items():
                    f.write(f"{class_name}: {count}\n")
                f.write("\n===== OBJECT COUNTS AND ACCURACY PER IMAGE =====\n\n")
                for img_path in sorted(all_class_counts.keys()):
                    f.write(f"Image: {img_path}\n")
                    f.write(f"Estimated Accuracy: {self.get_accuracy_percentage(all_confidences[img_path]):.1f}%\n")
                    f.write(f"Quality: {self.get_accuracy_label(all_confidences[img_path])}\n")
                    f.write("Objects:\n")
                    for class_name, count in all_class_counts[img_path].items():
                        f.write(f"  {class_name}: {count}\n")
                    f.write("\n")
        
        return results, all_class_counts, all_confidences
    
    def process_multiple_images(self, image_paths, output_dir=None, visualize=True, save_results=True):
        """
        Process multiple images.
        
        Args:
            image_paths (list): List of paths to input images
            output_dir (str, optional): Path to save output images
            visualize (bool): Whether to visualize detections
            save_results (bool): Whether to save detection results to a text file
            
        Returns:
            dict: Dictionary mapping image paths to lists of detections
            dict: Dictionary mapping image paths to class counts
            dict: Dictionary mapping image paths to average confidence scores
        """
        # Create output directory if not exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process each image
        results = {}
        all_class_counts = {}
        all_confidences = {}
        total_counts = Counter()
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            img_path = Path(img_path)
            # Determine output path
            if output_dir:
                out_img_path = Path(output_dir) / img_path.name
            else:
                out_img_path = None
            
            # Process the image
            detections, class_counts, avg_confidence = self.process_image(
                str(img_path),
                str(out_img_path) if out_img_path else None,
                visualize
            )
            
            # Store results
            results[str(img_path)] = detections
            all_class_counts[str(img_path)] = class_counts
            all_confidences[str(img_path)] = avg_confidence
            
            # Update total counts
            total_counts.update(class_counts)
            
            # Save results to a text file if requested
            if save_results and output_dir:
                txt_path = out_img_path.with_suffix('.txt')
                with open(txt_path, 'w') as f:
                    f.write(f"Objects detected in {img_path.name}:\n")
                    for class_name, count in class_counts.items():
                        f.write(f"{class_name}: {count}\n")
                    f.write(f"\nEstimated Detection Accuracy: {self.get_accuracy_percentage(avg_confidence):.1f}%\n")
                    f.write(f"Quality: {self.get_accuracy_label(avg_confidence)}\n")
                    f.write("\nDetailed detections:\n")
                    for det in detections:
                        # Format: class_name confidence x y width height
                        f.write(f"{det[0]} {det[1]:.6f} {det[2]} {det[3]} {det[4]} {det[5]}\n")
        
        # Calculate overall average confidence
        overall_avg_confidence = sum(all_confidences.values()) / len(all_confidences) if all_confidences else 0.0
        
        # Create a summary of all class counts and accuracy
        if output_dir and save_results:
            with open(os.path.join(output_dir, "detection_summary.txt"), 'w') as f:
                f.write("===== DETECTION SUMMARY =====\n\n")
                f.write(f"Overall Estimated Accuracy: {self.get_accuracy_percentage(overall_avg_confidence):.1f}%\n")
                f.write(f"Detection Quality: {self.get_accuracy_label(overall_avg_confidence)}\n")
                f.write("\n===== TOTAL OBJECT COUNTS ACROSS ALL IMAGES =====\n")
                for class_name, count in total_counts.items():
                    f.write(f"{class_name}: {count}\n")
                f.write("\n===== OBJECT COUNTS AND ACCURACY PER IMAGE =====\n\n")
                for img_path in sorted(all_class_counts.keys()):
                    f.write(f"Image: {img_path}\n")
                    f.write(f"Estimated Accuracy: {self.get_accuracy_percentage(all_confidences[img_path]):.1f}%\n")
                    f.write(f"Quality: {self.get_accuracy_label(all_confidences[img_path])}\n")
                    f.write("Objects:\n")
                    for class_name, count in all_class_counts[img_path].items():
                        f.write(f"  {class_name}: {count}\n")
                    f.write("\n")
        
        return results, all_class_counts, all_confidences
    
    def export_results(self, results, all_confidences, output_path):
        """
        Export detection results to a text file.
        
        Args:
            results (dict): Dictionary mapping image paths to lists of detections
            all_confidences (dict): Dictionary mapping image paths to average confidence scores
            output_path (str): Path to save the results
        """
        with open(output_path, 'w') as f:
            # Calculate overall average confidence
            overall_avg_confidence = sum(all_confidences.values()) / len(all_confidences) if all_confidences else 0.0
            
            f.write("===== DETECTION SUMMARY =====\n\n")
            f.write(f"Overall Estimated Accuracy: {self.get_accuracy_percentage(overall_avg_confidence):.1f}%\n")
            f.write(f"Detection Quality: {self.get_accuracy_label(overall_avg_confidence)}\n\n")
            
            for img_path, detections in results.items():
                f.write(f"Image: {img_path}\n")
                avg_confidence = all_confidences.get(img_path, 0.0)
                f.write(f"Estimated Accuracy: {self.get_accuracy_percentage(avg_confidence):.1f}%\n")
                f.write(f"Quality: {self.get_accuracy_label(avg_confidence)}\n")
                
                # Count objects per class in this image
                class_counts = Counter([det[0] for det in detections])
                f.write("Object counts:\n")
                for class_name, count in class_counts.items():
                    f.write(f"  {class_name}: {count}\n")
                
                f.write("\nDetailed detections:\n")
                for det in detections:
                    class_name, confidence, x, y, width, height = det
                    f.write(f"  {class_name}: conf={confidence:.2f}, bbox=[{x}, {y}, {width}, {height}]\n")
                f.write("\n")


def is_image_file(path):
    """Check if a file is an image based on its extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    return Path(path).suffix.lower() in image_extensions


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection Pipeline")
    parser.add_argument("--model", required=True, help="Path to the YOLOv8 model weights (.pt file)")
    parser.add_argument("--input", required=True, nargs='+', help="Path to input directory or image file(s)")
    parser.add_argument("--output", default="output", help="Path to output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--no-visualize", action="store_true", help="Don't visualize detections")
    parser.add_argument("--no-save-results", action="store_true", help="Don't save detection results to text files")
    
    args = parser.parse_args()
    
    # Create the pipeline
    pipeline = YOLOv8DetectionPipeline(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        img_size=args.img_size
    )
    
    # Process the input(s)
    start_time = time.time()
    
    # Initialize results and class_counts
    results = {}
    class_counts = {}
    all_confidences = {}
    
    # Check if inputs are directories, single image, or multiple images
    input_paths = args.input
    image_paths = []
    dir_paths = []
    
    # Categorize inputs into directories and images
    for input_path in input_paths:
        path = Path(input_path)
        if path.is_dir():
            dir_paths.append(path)
        elif is_image_file(path):
            image_paths.append(path)
        else:
            print(f"Warning: {input_path} is not a valid directory or image file. Skipping.")
    
    # Process directories
    for dir_path in dir_paths:
        print(f"Processing directory: {dir_path}")
        dir_results, dir_class_counts, dir_confidences = pipeline.process_directory(
            input_dir=str(dir_path),
            output_dir=args.output,
            visualize=not args.no_visualize,
            save_results=not args.no_save_results
        )
        results.update(dir_results)
        class_counts.update(dir_class_counts)
        all_confidences.update(dir_confidences)
    
    # Process individual images
    if image_paths:
        print(f"Processing {len(image_paths)} individual image(s)")
        img_results, img_class_counts, img_confidences = pipeline.process_multiple_images(
            image_paths=image_paths,
            output_dir=args.output,
            visualize=not args.no_visualize,
            save_results=not args.no_save_results
        )
        results.update(img_results)
        class_counts.update(img_class_counts)
        all_confidences.update(img_confidences)
    
    elapsed_time = time.time() - start_time
    
    # Export combined results
    if results and not args.no_save_results:
        pipeline.export_results(results, all_confidences, os.path.join(args.output, "detections_summary.txt"))
    
    # Calculate overall average confidence
    overall_avg_confidence = sum(all_confidences.values()) / len(all_confidences) if all_confidences else 0.0
    
    # Print summary
    print(f"Processed {len(results)} images in {elapsed_time:.2f} seconds")
    print(f"Overall Estimated Accuracy: {pipeline.get_accuracy_percentage(overall_avg_confidence):.1f}%")
    print(f"Detection Quality: {pipeline.get_accuracy_label(overall_avg_confidence)}")
    
    # Calculate and print total objects detected
    total_counts = Counter()
    for counts in class_counts.values():
        total_counts.update(counts)
    
    if total_counts:
        print(f"Total objects detected by class:")
        for class_name, count in total_counts.items():
            print(f"  {class_name}: {count}")
    else:
        print("No objects detected.")
    
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()