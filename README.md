Real vs. AI-Generated Receipt Detection 🧾🤖
Author: Guntaka Pranav Nadh Reddy
Track: Software Engineering — Internship Project
Organization: Accenture — SAP Expense Enhancement Initiative
📌 Project Overview
This repository contains the deep learning pipeline developed to evaluate and classify genuine, physically captured receipt images versus synthetically generated (AI-rendered) receipts. The primary objective of this model is to support expense-fraud risk reduction within enterprise systems.
During development, the project heavily utilized Explainable AI (Grad-CAM) to diagnose "shortcut learning," leading to the engineering of a highly robust, adversarial data augmentation pipeline that forces the model to evaluate genuine structural authenticity rather than superficial image artifacts.
📊 Dataset Description
The dataset was structured as a binary classification corpus (Real vs. AI-Generated).
• Real Receipts Data Sources: SOIRE, WildReceipt, and expressExpense (sourced from Kaggle).
• AI Receipts Data Sources: GPT-4o fake receipts and DALL-E generated images.
Final Dataset Statistics
After cleaning, resizing all images to 448 × 448 pixels using LANCZOS resampling, standardizing to JPEG formats, and applying data augmentation to balance the minority class, the dataset consisted of:
Class	Image Count	Unique Source Groups	Class Weight Applied
Real Receipts	1,950	1,075	0.9062
AI-Generated Receipts	1,584	808	1.1155
Total	3,534	1,883	—
Data Leakage Prevention: To prevent data leakage, the dataset was split at the "group level" so that an original image and its augmented siblings were always kept in the exact same partition. The split was exactly 70% Train, 14.9% Validation, and 15.1% Test.
🛑 The "Shortcut Learning" Discovery
During the initial training of the ResNet50 architecture, the model converged suspiciously fast, achieving >99% validation accuracy within the first two epochs. Because real-world document forgery detection rarely hits this metric so quickly, it was treated as a diagnostic red flag.
Using Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the network's attention layers, three distinct failure modes ("shortcuts") were identified:
1. Background Context Bias (Mode A): The model completely ignored the receipt text and looked at the background. It learned that textured backgrounds (like wood) meant "Real", while plain backgrounds meant "AI".
2. Capture-Hardware Artifact Bias (Mode B): The model acted as a camera-type classifier. It associated uniform, even lighting with AI renders, and flash glares/blur with real smartphones.
3. Macro-Layout Shortcut (Mode C): The model memorized the overall structural formatting of the templates rather than looking for fine-grained AI text irregularities.
🛠️ The Remediation Pipeline (Anti-Shortcut Augmentation)
To neutralize these shortcuts, an aggressive, targeted data augmentation pipeline was engineered in PyTorch. By artificially degrading both classes during training, the model was forced to evaluate actual receipt content.
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomRotation(degrees=45, fill=255),                      # Breaks background & framing bias
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), # Kills lighting/capture hardware bias
    T.GaussianBlur(kernel_size=5),                               # Kills render-sharpness bias
    T.ToTensor(),
    T.RandomErasing(p=0.3, scale=(0.02, 0.1)),                   # Breaks macro-layout memorization
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

🚀 Final Validated Results
Following the deployment of the anti-shortcut pipeline, the artificial 99% accuracy collapsed into a much healthier, trustworthy learning trajectory.
• Final Test Accuracy: 93.0%
• Validation Methodology: Group-isolated, leakage-verified split
• Shortcut Status: Mitigated via anti-shortcut augmentation
A drop in headline accuracy from over 99% to 93% represented an improvement in actual model quality, as the remediated model can now generalize to new receipts in the real world without relying on flawed background or lighting cues.
🧠 Key Learnings
• Early High Accuracy is a Warning: Achieving near-perfect validation accuracy in 1-2 epochs on a complex forgery task should be treated as a diagnostic signal for data shortcuts, not an immediate success.
• Run XAI Early: Explainable AI tools like Grad-CAM should be integrated early in the development cycle to catch flawed learning before significant time is invested.
• Targeted Augmentation as a Cure: Aggressive augmentation can be used deliberately to selectively destroy specific signal shortcuts, forcing the neural network to optimize toward genuine task features.
