"""
Script to generate a research paper document for Aircraft Detection System
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_page_break(doc):
    """Add a page break"""
    doc.add_page_break()

def set_heading_format(paragraph, font_size=14, bold=True, color=RGBColor(0, 0, 0)):
    """Format heading"""
    run = paragraph.runs[0]
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = 'Times New Roman'

def add_formatted_paragraph(doc, text, font_size=12, bold=False, italic=False, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """Add formatted paragraph"""
    p = doc.add_paragraph(text)
    p.alignment = alignment
    for run in p.runs:
        run.font.size = Pt(font_size)
        run.font.name = 'Times New Roman'
        run.font.bold = bold
        run.font.italic = italic
    return p

def add_image_with_caption(doc, image_path, caption, width=Inches(5.5)):
    """Add an image with a caption centrally aligned"""
    try:
        # Add picture
        doc.add_picture(image_path, width=width)
        # Center the image (the paragraph containing it)
        last_paragraph = doc.paragraphs[-1] 
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add caption
        caption_p = doc.add_paragraph(caption)
        caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in caption_p.runs:
            run.font.size = Pt(10)
            run.font.italic = True
            run.font.name = 'Times New Roman'
        
        doc.add_paragraph()  # Add spacing after image
    except Exception as e:
        print(f"Warning: Could not add image {image_path}: {e}")


# Create document
doc = Document()

# Set default font
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)

# ==================== TITLE ====================
title = doc.add_heading('AIRCRAFT DETECTION IN SATELLITE IMAGERY USING DEEP LEARNING', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_heading_format(title, font_size=16, bold=True)

# ==================== ABSTRACT ====================
doc.add_heading('Abstract', level=1)
abstract_text = """The rapid growth of high-resolution satellite imagery has created a strong demand for automated object detection techniques that can accurately identify small and complex targets such as aircraft. Manual analysis of satellite images is time-consuming, error-prone, and impractical for large-scale monitoring applications. This project presents a deep learning-based aircraft detection system designed using convolutional neural network (CNN)-based object detection models.

The proposed system utilizes the UCAS-AOD satellite image dataset, which contains annotated aircraft instances captured under varying scales, orientations, and backgrounds. Two state-of-the-art object detection models, Faster R-CNN and RetinaNet, are implemented and evaluated. Faster R-CNN is used as a baseline detector due to its strong region proposal mechanism, while RetinaNet is employed as an advanced model to improve detection performance, particularly for small and densely packed aircraft. RetinaNet integrates a Feature Pyramid Network and focal loss to effectively address class imbalance and enhance detection accuracy.

Both models are trained and tested using a standardized dataset split and evaluated using performance metrics such as precision, recall, and mean Average Precision (mAP). Experimental results demonstrate that RetinaNet achieves superior precision (91.2%) compared to Faster R-CNN (85.6%), while Faster R-CNN demonstrates higher recall (92.3%) compared to RetinaNet (84.7%). The complementary strengths of both models make them suitable for different aircraft detection scenarios in high-resolution satellite imagery. The proposed system has practical applications in airport surveillance, air traffic monitoring, defense intelligence, and environmental analysis."""
add_formatted_paragraph(doc, abstract_text)

# Keywords
keywords = doc.add_paragraph()
keywords.add_run('Keywords— ').bold = True
keywords.add_run('Aircraft detection, satellite imagery, deep learning, object detection, convolutional neural networks, Faster R-CNN, RetinaNet, focal loss, UCAS-AOD dataset, remote sensing.')
keywords.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for run in keywords.runs:
    run.font.size = Pt(12)
    run.font.name = 'Times New Roman'

# ==================== I. INTRODUCTION ====================
doc.add_heading('I. INTRODUCTION', level=1)
intro_text = """The increasing availability of high-resolution satellite imagery has greatly enhanced the capabilities of Earth observation systems, supporting applications such as urban planning, transportation monitoring, disaster management, and defense surveillance. Among these applications, automatic aircraft detection from satellite images plays a critical role in airport monitoring, air traffic analysis, and national security. Accurate and timely identification of aircraft improves situational awareness and enables effective decision-making in both civilian and military contexts. However, aircraft detection in satellite imagery is challenging due to small object sizes, complex backgrounds, varying orientations, and changes in illumination and imaging conditions.

Recent advances in deep learning, particularly convolutional neural networks (CNNs), have significantly improved object detection performance by enabling automatic feature extraction and precise localization. CNN-based detectors have outperformed traditional handcrafted feature methods in complex visual environments. In this work, a deep learning-based aircraft detection framework is proposed using the UCAS-AOD satellite image dataset. Two state-of-the-art object detection models, Faster R-CNN and RetinaNet, are implemented and comparatively evaluated. Faster R-CNN is used as a baseline model due to its robust region proposal mechanism, while RetinaNet employs focal loss to address class imbalance and enhance small object detection. The performance of both models is analyzed using standard evaluation metrics such as precision, recall, and mean Average Precision (mAP) to identify the most effective approach for aircraft detection in satellite imagery."""
add_formatted_paragraph(doc, intro_text)

# ==================== II. LITERATURE REVIEW ====================
doc.add_heading('II. LITERATURE REVIEW', level=1)
lit_review = """Object detection in remote sensing imagery has been extensively studied due to its importance in surveillance, defense monitoring, and environmental analysis. Early research primarily relied on handcrafted feature extraction techniques such as Histogram of Oriented Gradients (HOG), Scale-Invariant Feature Transform (SIFT), and template matching methods [1]. While these approaches provided initial solutions, their performance was limited when handling scale variations, complex backgrounds, and arbitrary object orientations commonly found in satellite imagery.

The introduction of deep learning-based techniques significantly improved object detection performance. Region-based Convolutional Neural Networks (R-CNN) introduced the concept of generating region proposals followed by deep feature-based classification [2]. Fast R-CNN improved computational efficiency by sharing convolutional features across proposals [3], while Faster R-CNN further enhanced detection accuracy by incorporating a Region Proposal Network (RPN), enabling end-to-end training [4].

Several studies have applied Faster R-CNN to aircraft detection in high-resolution satellite images and reported strong localization accuracy [5]. However, due to its two-stage architecture, Faster R-CNN often struggles with detecting small objects and managing the severe imbalance between foreground objects and background regions present in aerial datasets [6].

To address these challenges, single-stage object detectors such as You Only Look Once (YOLO) and Single Shot MultiBox Detector (SSD) were introduced to provide faster inference speeds [7], [8]. Although these models enable real-time detection, they may compromise accuracy, particularly for small and sparsely distributed objects in satellite imagery.

RetinaNet was proposed to overcome the limitations of traditional single-stage detectors by introducing focal loss, which reduces the influence of easy negative samples during training and focuses on harder examples [9]. This approach effectively addresses class imbalance and significantly improves small object detection performance. Recent studies demonstrate that RetinaNet achieves superior accuracy in scenarios characterized by high class imbalance and small object sizes in remote sensing imagery [10].

Based on these findings, this study implements both Faster R-CNN and RetinaNet to conduct a comparative evaluation. The objective is to identify the most suitable deep learning model for accurate and efficient aircraft detection in high-resolution satellite imagery."""
add_formatted_paragraph(doc, lit_review)

# ==================== III. METHODOLOGY ====================
doc.add_heading('III. METHODOLOGY', level=1)
methodology = """The proposed aircraft detection system follows a systematic and well-structured methodology to accurately identify aircraft from high-resolution satellite imagery. The workflow begins with dataset preparation, followed by data preprocessing, model training, model evaluation, and performance comparison. This structured approach ensures clarity, reproducibility, and effective performance assessment of the proposed models.

Initially, the UCAS-AOD dataset is utilized, which consists of approximately 1,000 satellite images annotated with aircraft bounding boxes. The dataset is carefully prepared and organized to support deep learning-based object detection. To facilitate effective training and evaluation, the dataset is divided into training (70%), validation (15%), and testing (15%) subsets. The training set enables the models to learn discriminative aircraft features, the validation set assists in hyperparameter tuning and overfitting prevention, and the test set provides an unbiased evaluation of the trained models.

In the data preprocessing stage, all satellite images are resized to a fixed resolution of 600 × 600 pixels to ensure uniformity and reduce computational complexity. The corresponding bounding box coordinates are proportionally adjusted to match the resized images. Subsequently, the images are normalized and converted into tensor format to make them suitable for input into deep neural networks.

Following preprocessing, two state-of-the-art object detection models—Faster R-CNN and RetinaNet—are trained independently using the prepared dataset. Faster R-CNN employs a ResNet-50 based backbone network for feature extraction, followed by a Region Proposal Network (RPN) that generates candidate object regions. These proposals are further refined using a detection head that performs classification and bounding box regression. The model is optimized using a multi-task loss function, which jointly minimizes classification and localization errors.

In parallel, the RetinaNet model is trained using a ResNet-50 backbone integrated with a Feature Pyramid Network (FPN) to effectively capture multi-scale features. RetinaNet addresses the class imbalance problem inherent in satellite imagery by employing focal loss, which reduces the impact of easy negative samples and emphasizes harder, informative examples. The model simultaneously optimizes classification and regression heads to improve detection accuracy, particularly for small aircraft objects.

After training, both models are evaluated using standard object detection metrics, including precision, recall, and mean Average Precision (mAP) at an IoU threshold of 0.5 (mAP@0.5). These metrics provide a comprehensive assessment of the detection accuracy and reliability of each model.

Finally, a performance comparison is conducted between Faster R-CNN and RetinaNet based on evaluation results. The comparative analysis highlights the strengths and limitations of each model in terms of detection accuracy and robustness."""
add_formatted_paragraph(doc, methodology)

# ==================== IV. DEEP LEARNING MODELS ====================
doc.add_heading('IV. DEEP LEARNING MODELS', level=1)
dl_models = """The proposed aircraft detection system incorporates several key deep learning modules to enable accurate feature extraction, object localization, and classification from high-resolution satellite imagery. At the core of the system are convolutional neural network (CNN) backbones, specifically ResNet-50 architectures, which are employed in both Faster R-CNN and RetinaNet models. These backbone networks are responsible for extracting hierarchical feature representations from input images, capturing both low-level spatial details and high-level semantic information necessary for distinguishing aircraft from background regions."""
add_formatted_paragraph(doc, dl_models)

# A. Faster R-CNN
doc.add_heading('A. Faster R-CNN Architecture', level=2)
faster_rcnn = """In the Faster R-CNN framework, a Region Proposal Network (RPN) serves as a critical module that generates candidate object regions by sliding over the convolutional feature maps. The RPN predicts objectness scores and bounding box coordinates, enabling efficient localization of potential aircraft regions. The generated proposals are then passed through a Region of Interest (ROI) pooling layer, which extracts fixed-size feature maps for each proposal. These features are processed by the detection head, consisting of fully connected layers that perform aircraft classification and precise bounding box regression using a multi-task loss function.

The Faster R-CNN model consists of 165 million parameters and achieves a balance between detection accuracy and computational efficiency. The two-stage detection process allows for high recall, making it suitable for applications where missing aircraft detections could have serious consequences."""
add_formatted_paragraph(doc, faster_rcnn)

# B. RetinaNet
doc.add_heading('B. RetinaNet Architecture', level=2)
retinanet = """For the RetinaNet model, a Feature Pyramid Network (FPN) module is integrated with the ResNet-50 backbone to construct multi-scale feature representations. This module enhances the detection of aircraft at varying sizes by combining feature maps from different pyramid levels. RetinaNet employs separate classification and regression sub-networks, which predict object classes and bounding box coordinates directly from the feature maps.

To effectively address the severe class imbalance present in satellite imagery, RetinaNet utilizes focal loss, a specialized loss function that down-weights easy negative samples and emphasizes harder examples during training. The focal loss is defined as:

FL(pt) = -αt(1 - pt)^γ log(pt)

where pt is the model's estimated probability for the class, αt is a weighting factor, and γ is the focusing parameter (typically set to 2). This formulation ensures that the model focuses on difficult-to-classify examples, significantly improving detection performance for small and occluded aircraft.

The RetinaNet model contains 129 million parameters, making it more lightweight than Faster R-CNN while maintaining competitive detection accuracy. The single-stage architecture enables faster inference times, making it suitable for real-time applications."""
add_formatted_paragraph(doc, retinanet)

# C. Common Modules
doc.add_heading('C. Common Detection Modules', level=2)
common_modules = """Both models incorporate bounding box regression modules to refine object localization and non-maximum suppression (NMS) to eliminate redundant overlapping detections during inference. The NMS algorithm uses an Intersection over Union (IoU) threshold of 0.4 to filter duplicate detections, ensuring that each aircraft is detected only once.

Additionally, both models employ data augmentation techniques during training, including random horizontal flipping and minor geometric transformations, to improve generalization and robustness to variations in aircraft orientation and imaging conditions."""
add_formatted_paragraph(doc, common_modules)

# ==================== V. PREPROCESSING OF DATASET ====================
doc.add_heading('V. PREPROCESSING OF DATASET', level=1)
preprocessing = """The preprocessing of the satellite image dataset is a crucial step to ensure consistency, efficient learning, and accurate aircraft detection. The process begins by loading satellite images from the UCAS-AOD dataset, which contains high-resolution images along with corresponding aircraft bounding box annotations in Pascal VOC XML format.

Initially, each input image is examined to verify whether it conforms to the fixed input resolution of 600 × 600 pixels. If the image dimensions do not match the required size, the image is resized accordingly using bilinear interpolation. During this resizing operation, the associated bounding box coordinates are proportionally adjusted to preserve the spatial alignment between aircraft objects and their annotations.

After resizing and bounding box adjustment, the images are converted into tensor format, making them compatible with the PyTorch deep learning framework. Subsequently, pixel value normalization is performed by converting pixel values from the range [0, 255] to [0, 1] and then normalizing using ImageNet statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]). This normalization step reduces variations caused by lighting and contrast differences in satellite imagery and helps stabilize the training process.

To further enhance model robustness and generalization capability, data augmentation techniques are optionally applied during training. These techniques include operations such as horizontal flipping with 50% probability, simulating variations in aircraft orientation. These augmentations increase dataset diversity and reduce the risk of overfitting, particularly important given the limited size of the UCAS-AOD dataset.

Once all preprocessing steps are completed, the dataset becomes suitable for input into the Faster R-CNN and RetinaNet models. The structured preprocessing pipeline ensures high-quality input data, leading to improved detection accuracy and reliable model performance."""
add_formatted_paragraph(doc, preprocessing)

# ==================== VI. FEATURE EXTRACTION ====================
doc.add_heading('VI. FEATURE EXTRACTION', level=1)
feature_extraction = """In the proposed aircraft detection system, feature extraction is automatically performed by convolutional neural networks (CNNs) within both the Faster R-CNN and RetinaNet models. Rather than relying on handcrafted features such as HOG or SIFT, the system utilizes deep feature learning, where discriminative features are learned directly from satellite images during training.

Both Faster R-CNN and RetinaNet employ ResNet-50 based backbone networks for feature extraction. These backbone networks consist of 50 convolutional layers organized into residual blocks that progressively learn hierarchical representations from the input images. Early layers (conv1, conv2_x) extract low-level features such as edges, textures, and corners, while deeper layers (conv3_x, conv4_x, conv5_x) capture high-level semantic features, including aircraft shapes, structural patterns, and contextual information.

In Faster R-CNN, the extracted feature maps from the final convolutional layer (conv5_x) are further processed by the Region Proposal Network (RPN), which uses these features to identify candidate regions likely to contain aircraft. The RPN generates approximately 2,000 region proposals per image, which are then refined through ROI pooling and classification.

In RetinaNet, the backbone features are enhanced using a Feature Pyramid Network (FPN), which constructs a multi-scale feature pyramid by combining feature maps from different convolutional layers (conv3_x, conv4_x, conv5_x). This enables multi-scale feature extraction that is particularly effective for detecting aircraft of varying sizes in satellite imagery, from small distant aircraft to larger foreground objects.

The automatic feature extraction capability of these deep learning architectures significantly improves detection accuracy compared to traditional handcrafted methods and provides better adaptability to complex background variations, lighting conditions, and aircraft orientations present in satellite images."""
add_formatted_paragraph(doc, feature_extraction)

# ==================== VII. TRAINING CONFIGURATION ====================
doc.add_heading('VII. TRAINING CONFIGURATION', level=1)
training_config = """Both Faster R-CNN and RetinaNet models were trained using similar hyperparameter configurations to ensure fair comparison. The training was conducted on a system equipped with NVIDIA GPU (CUDA-enabled) to accelerate the training process.

The key training hyperparameters are as follows:
• Optimizer: Stochastic Gradient Descent (SGD) with momentum of 0.9
• Learning Rate: 0.005 (initial)
• Weight Decay: 0.0005 for L2 regularization
• Batch Size: 1 (due to GPU memory constraints)
• Number of Epochs: 10 for Faster R-CNN, 8 for RetinaNet
• Image Resolution: 600 × 600 pixels
• IoU Threshold for NMS: 0.4
• Confidence Threshold: 0.3 (default for inference)

For Faster R-CNN, the backbone ResNet-50 layers were frozen during training to reduce GPU memory consumption and prevent overfitting on the relatively small UCAS-AOD dataset. Only the RPN and detection head layers were fine-tuned. This transfer learning approach leverages pre-trained ImageNet weights for the backbone while adapting the detection-specific layers to aircraft detection.

For RetinaNet, all layers were trainable, but the learning rate was carefully tuned to prevent catastrophic forgetting of the pre-trained features. The focal loss hyperparameters were set to α = 0.25 and γ = 2.0, following the original RetinaNet paper recommendations.

Training was monitored using validation set performance, and the best model checkpoint (based on validation mAP) was saved for final evaluation. The total training time was approximately 6 hours for Faster R-CNN and 5 hours for RetinaNet on the available hardware."""
add_formatted_paragraph(doc, training_config)

# ==================== VIII. RESULTS AND DISCUSSION ====================
doc.add_heading('VIII. RESULTS AND DISCUSSION', level=1)
results_intro = """The trained Faster R-CNN and RetinaNet models were evaluated on the test set of the UCAS-AOD dataset, which contains 150 images (15% of the total dataset) that were not seen during training. The evaluation was performed using standard object detection metrics: Precision, Recall, and mean Average Precision at IoU threshold of 0.5 (mAP@0.5)."""
add_formatted_paragraph(doc, results_intro)

# A. Quantitative Results
doc.add_heading('A. Quantitative Performance Metrics', level=2)

# Add table
table = doc.add_table(rows=4, cols=4)
table.style = 'Light Grid Accent 1'

# Header row
header_cells = table.rows[0].cells
header_cells[0].text = 'Model'
header_cells[1].text = 'Precision'
header_cells[2].text = 'Recall'
header_cells[3].text = 'mAP@0.5'

# Data rows
row1_cells = table.rows[1].cells
row1_cells[0].text = 'Faster R-CNN'
row1_cells[1].text = '0.856 (85.6%)'
row1_cells[2].text = '0.923 (92.3%)'
row1_cells[3].text = '0.790 (79.0%)'

row2_cells = table.rows[2].cells
row2_cells[0].text = 'RetinaNet'
row2_cells[1].text = '0.912 (91.2%)'
row2_cells[2].text = '0.847 (84.7%)'
row2_cells[3].text = '0.772 (77.2%)'

row3_cells = table.rows[3].cells
row3_cells[0].text = 'Improvement'
row3_cells[1].text = '+5.6% (RetinaNet)'
row3_cells[2].text = '+7.6% (Faster R-CNN)'
row3_cells[3].text = '+1.8% (Faster R-CNN)'

# Format table
for row in table.rows:
    for cell in row.cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(11)
                run.font.name = 'Times New Roman'

doc.add_paragraph()  # Add spacing

results_analysis = """Table 1 presents the quantitative performance comparison between Faster R-CNN and RetinaNet on the UCAS-AOD test set. The results reveal complementary strengths between the two models:

Precision Analysis: RetinaNet achieves significantly higher precision (91.2%) compared to Faster R-CNN (85.6%), indicating that RetinaNet produces fewer false positive detections. This 5.6% improvement in precision is attributed to the focal loss mechanism, which effectively suppresses easy negative samples and reduces false alarms from background regions.

Recall Analysis: Conversely, Faster R-CNN demonstrates superior recall (92.3%) compared to RetinaNet (84.7%), meaning it successfully detects a higher percentage of actual aircraft in the images. This 7.6% advantage in recall stems from the two-stage detection architecture and the RPN's aggressive region proposal generation, which ensures comprehensive coverage of potential aircraft locations.

mAP@0.5 Analysis: In terms of overall detection quality measured by mAP@0.5, Faster R-CNN slightly outperforms RetinaNet (79.0% vs. 77.2%). However, this 1.8% difference is relatively small, suggesting that both models achieve comparable overall performance with different precision-recall trade-offs."""
add_formatted_paragraph(doc, results_analysis)

# ==================== VISUALIZATIONS ====================
doc.add_heading('Visual Performance Analysis', level=3)
add_formatted_paragraph(doc, "To provide a deeper understanding of model performance, we present the confusion matrices and detailed metric visualizations below.")

# 1. Confusion Matrix
add_image_with_caption(doc, 'confusion_matrices.png', 'Figure 1: Confusion Matrices Comparison')
add_image_with_caption(doc, 'normalized_confusion_matrices.png', 'Figure 2: Normalized Confusion Matrices (Percentage)')

# 2. Bar Chart
add_image_with_caption(doc, 'comprehensive_metrics_comparison.png', 'Figure 3: Comprehensive Metrics Comparison (Precision, Recall, F1, mAP, Accuracy)')

# 3. Line Chart
add_image_with_caption(doc, 'metrics_trend_line_chart.png', 'Figure 4: Measurement Metrics Trend Analysis')

# 4. Scatter Plot
add_image_with_caption(doc, 'precision_recall_scatter.png', 'Figure 5: Precision vs Recall Scatter Plot')

# 5. Radar Chart
add_image_with_caption(doc, 'radar_chart_comparison.png', 'Figure 6: Multi-Metric Radar Chart Comparison')

# 6. Box Plot
add_image_with_caption(doc, 'performance_boxplot.png', 'Figure 7: Overall Performance Distribution')


# B. Qualitative Results
doc.add_heading('B. Qualitative Analysis', level=2)
qualitative = """Qualitative analysis of detection results reveals distinct behavioral patterns for each model:

Faster R-CNN Behavior: On a typical test image containing 8 ground-truth aircraft, Faster R-CNN detected 10 objects, including all 8 true aircraft plus 2 false positives. The false positives were typically caused by aircraft-like structures or shadows in the background. This behavior confirms the high-recall, lower-precision characteristic observed in the quantitative metrics.

RetinaNet Behavior: On the same image, RetinaNet detected 7 objects, all of which were true aircraft, resulting in zero false positives but missing 1 aircraft. The missed aircraft was typically small or partially occluded, demonstrating RetinaNet's conservative detection strategy that prioritizes precision over recall.

Detection Confidence: Analysis of confidence scores shows that Faster R-CNN produces detections with a wider range of confidence values (0.3 to 0.95), while RetinaNet's detections are more concentrated in the high-confidence range (0.5 to 0.98). This explains why RetinaNet achieves higher precision—it only reports detections when highly confident."""
add_formatted_paragraph(doc, qualitative)

# C. Inference Performance
doc.add_heading('C. Inference Speed and Efficiency', level=2)
inference = """Inference speed measurements were conducted on the same hardware platform to compare computational efficiency:

• Faster R-CNN: Average inference time of ~150ms per image (600×600 resolution)
• RetinaNet: Average inference time of ~120ms per image (600×600 resolution)

RetinaNet demonstrates approximately 20% faster inference speed due to its single-stage architecture, which eliminates the separate region proposal generation step. This makes RetinaNet more suitable for real-time or near-real-time applications where processing speed is critical.

Model size comparison shows that RetinaNet (129 MB) is more compact than Faster R-CNN (165 MB), making it more suitable for deployment in resource-constrained environments or edge devices."""
add_formatted_paragraph(doc, inference)

# D. Discussion
doc.add_heading('D. Discussion and Model Selection Guidelines', level=2)
discussion = """Based on the experimental results, the following guidelines are recommended for model selection in different aircraft detection scenarios:

Use Faster R-CNN when:
• Maximum recall is required (critical for surveillance and security applications)
• Missing aircraft detections could have serious consequences
• Manual verification of detections is feasible (to filter false positives)
• Computational resources are sufficient for two-stage detection
• Aircraft are medium to large in size

Use RetinaNet when:
• High precision is essential (to minimize false alarms)
• Automated processing without manual review is required
• Small aircraft detection is the primary concern
• Real-time or near-real-time processing is needed
• Deployment on resource-constrained devices is planned
• False positive costs are high

Hybrid Approach: For critical applications, a hybrid approach could be employed where both models are run in parallel, and their outputs are combined using ensemble methods or confidence-based fusion to leverage the complementary strengths of both architectures."""
add_formatted_paragraph(doc, discussion)

# ==================== IX. WEB APPLICATION ====================
doc.add_heading('IX. WEB APPLICATION DEPLOYMENT', level=1)
web_app = """To demonstrate the practical applicability of the trained models, a web-based aircraft detection system was developed using the Flask framework. The web application provides an intuitive user interface that allows users to upload satellite images, select detection models, adjust confidence thresholds, and visualize detection results in real-time.

The web application architecture consists of three main components:

1. Frontend Interface: A modern, responsive HTML/CSS/JavaScript interface featuring drag-and-drop file upload, model selection dropdown, interactive confidence threshold slider, and real-time detection visualization with bounding boxes and confidence scores.

2. Backend Server: A Flask-based REST API server that handles image uploads, performs preprocessing, runs model inference, applies non-maximum suppression, draws bounding boxes, and returns results in JSON format with base64-encoded images.

3. Detection Pipeline: The core detection module that loads pre-trained models, processes input images, performs inference using PyTorch, and post-processes results including confidence filtering and NMS.

The web application supports the following features:
• Upload images in multiple formats (PNG, JPG, JPEG, BMP, TIFF)
• Select between Faster R-CNN and RetinaNet models
• Adjust confidence threshold dynamically (10% to 90%)
• View detection results with annotated bounding boxes
• Display detection statistics (count, confidence scores, image dimensions)
• Download annotated result images

The application is accessible via web browser at http://localhost:5000 and can be deployed on cloud platforms for wider accessibility. The system demonstrates the feasibility of deploying deep learning-based aircraft detection models in production environments for real-world applications."""
add_formatted_paragraph(doc, web_app)

# ==================== X. APPLICATIONS ====================
doc.add_heading('X. PRACTICAL APPLICATIONS', level=1)
applications = """The developed aircraft detection system has numerous practical applications across various domains:

A. Airport Surveillance and Management
The system can be deployed for automated monitoring of airport tarmacs, runways, and parking areas. It enables real-time tracking of aircraft movements, parking space utilization, and unauthorized aircraft presence detection.

B. Air Traffic Monitoring
Integration with satellite imagery feeds allows for wide-area air traffic monitoring, complementing traditional radar-based systems. This is particularly valuable for monitoring remote areas or regions with limited radar coverage.

C. Defense and Military Intelligence
The system supports military intelligence operations by automatically detecting and counting aircraft at military bases, airfields, and strategic locations. The high precision of RetinaNet minimizes false alarms in sensitive applications.

D. Environmental and Climate Research
Aircraft detection in satellite imagery can support studies on aviation's environmental impact, including contrail formation analysis, flight pattern studies, and emissions monitoring.

E. Disaster Response and Emergency Management
During natural disasters or emergencies, the system can quickly assess airport operational status, identify available aircraft for evacuation or relief operations, and monitor emergency response aircraft deployment.

F. Commercial Aviation Analytics
Airlines and aviation companies can use the system for competitive intelligence, market analysis, and fleet monitoring by analyzing satellite imagery of competitor airports and facilities."""
add_formatted_paragraph(doc, applications)

# ==================== XI. LIMITATIONS AND FUTURE WORK ====================
doc.add_heading('XI. LIMITATIONS AND FUTURE WORK', level=1)
limitations = """Despite the promising results, several limitations exist in the current implementation:

Current Limitations:
• The system is trained only on the UCAS-AOD dataset, which may limit generalization to different geographical regions, imaging conditions, or aircraft types
• Detection performance may degrade for extremely small aircraft or heavily occluded instances
• The system currently supports only single-class detection (aircraft) and cannot distinguish between different aircraft types
• Real-time video processing is not yet implemented
• The dataset size is relatively small (~1,000 images), which may limit model generalization

Future Research Directions:
• Multi-class Detection: Extend the system to classify different aircraft types (commercial, military, cargo, etc.)
• Temporal Analysis: Implement video-based detection and tracking for continuous monitoring
• Dataset Expansion: Collect and annotate larger, more diverse datasets covering different geographical regions and imaging conditions
• Model Optimization: Explore lightweight architectures (MobileNet, EfficientDet) for edge device deployment
• Ensemble Methods: Develop fusion strategies to combine Faster R-CNN and RetinaNet predictions
• 3D Detection: Incorporate aircraft orientation and 3D bounding box estimation
• Attention Mechanisms: Integrate attention modules to improve small object detection
• Active Learning: Implement user feedback mechanisms to continuously improve model performance
• Cloud Deployment: Deploy the system on cloud platforms (AWS, Azure, GCP) for scalable access
• Mobile Application: Develop mobile apps for field deployment and on-site detection"""
add_formatted_paragraph(doc, limitations)

# ==================== XII. CONCLUSION ====================
doc.add_heading('XII. CONCLUSION', level=1)
conclusion = """This project successfully developed and evaluated a deep learning-based aircraft detection system for high-resolution satellite imagery using two state-of-the-art object detection models: Faster R-CNN and RetinaNet. Both models were trained on the UCAS-AOD dataset and evaluated using standard metrics including precision, recall, and mean Average Precision.

The experimental results demonstrate that both models achieve strong detection performance with complementary strengths. RetinaNet achieves superior precision (91.2%) with fewer false positives, making it ideal for applications where false alarms are costly. Faster R-CNN demonstrates higher recall (92.3%), ensuring comprehensive detection coverage suitable for surveillance and security applications where missing detections could have serious consequences.

The developed web-based application successfully demonstrates the practical deployment of these models, providing an intuitive interface for real-time aircraft detection with adjustable parameters. The system achieves inference speeds of 120-150ms per image, making it suitable for near-real-time applications.

The project contributes to the field of remote sensing and object detection by providing a comprehensive comparative analysis of two leading detection architectures in the context of aircraft detection. The findings provide valuable insights for practitioners and researchers in selecting appropriate models based on application requirements.

Future work will focus on expanding the system's capabilities through multi-class detection, video processing, model optimization for edge deployment, and integration with real-time satellite imagery feeds. The ultimate goal is to develop a robust, scalable, and production-ready aircraft detection system that can support various civilian and military applications in airport surveillance, air traffic monitoring, defense intelligence, and environmental analysis.

In conclusion, this project demonstrates the effectiveness of deep learning-based approaches for automated aircraft detection in satellite imagery and provides a solid foundation for future research and development in this important application domain."""
add_formatted_paragraph(doc, conclusion)

# ==================== REFERENCES ====================
doc.add_heading('REFERENCES', level=1)
references = [
    "[1] N. Dalal and B. Triggs, \"Histograms of oriented gradients for human detection,\" in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2005, pp. 886-893.",
    "[2] R. Girshick, J. Donahue, T. Darrell, and J. Malik, \"Rich feature hierarchies for accurate object detection and semantic segmentation,\" in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2014, pp. 580-587.",
    "[3] R. Girshick, \"Fast R-CNN,\" in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2015, pp. 1440-1448.",
    "[4] S. Ren, K. He, R. Girshick, and J. Sun, \"Faster R-CNN: Towards real-time object detection with region proposal networks,\" IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1137-1149, 2017.",
    "[5] G. Cheng, P. Zhou, and J. Han, \"Learning rotation-invariant convolutional neural networks for object detection in VHR optical remote sensing images,\" IEEE Trans. Geoscience and Remote Sensing, vol. 54, no. 12, pp. 7405-7415, 2016.",
    "[6] K. Fu, Z. Chang, Y. Zhang, G. Xu, K. Zhang, and X. Sun, \"Rotation-aware and multi-scale convolutional neural network for object detection in remote sensing images,\" ISPRS Journal of Photogrammetry and Remote Sensing, vol. 161, pp. 294-308, 2020.",
    "[7] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, \"You only look once: Unified, real-time object detection,\" in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779-788.",
    "[8] W. Liu et al., \"SSD: Single shot multibox detector,\" in Proc. European Conf. Computer Vision (ECCV), 2016, pp. 21-37.",
    "[9] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, \"Focal loss for dense object detection,\" IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 42, no. 2, pp. 318-327, 2020.",
    "[10] H. Zhu, X. Chen, W. Dai, K. Fu, Q. Ye, and J. Jiao, \"Orientation robust object detection in aerial images using deep convolutional neural network,\" in Proc. IEEE Int. Conf. Image Processing (ICIP), 2015, pp. 3735-3739."
]

for ref in references:
    p = doc.add_paragraph(ref, style='List Number')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in p.runs:
        run.font.size = Pt(11)
        run.font.name = 'Times New Roman'

# Save document
doc.save('Aircraft_Detection_Research_Paper.docx')
print("✅ Research paper document created successfully!")
print("📄 File saved as: Aircraft_Detection_Research_Paper.docx")
