# BiLSTM Models for Sentiment Analysis


This repository contains the implementation of **BiLSTM models** for document-level sentiment analysis, comparing the performance of traditional and improved BiLSTM techniques with BERT. The work is based on your research focusing on **Amazon book review data**, showcasing enhanced techniques like fine-tuning, attention, pooling, and layer normalization to achieve state-of-the-art results.

## Project Overview

### Background
Sentiment analysis is a key area in natural language processing (NLP), aiming to evaluate emotions and opinions from digital content. This project explores document-level sentiment analysis, focusing on:
- Evaluating traditional BiLSTM models.
- Comparing improved BiLSTM models with BERT.
- Enhancing BiLSTM with fine-tuning, attention mechanisms, pooling, and layer normalization.

### Key Contributions
- **Enhanced BiLSTM model**: Incorporating fine-tuning, attention, and layer normalization to improve sentiment analysis performance.
- **Comparative Analysis**: Performance comparison between BERT and BiLSTM-based models using recall and F1 score.
- **Practical Applications**: Insights for applications in marketing, recommendation systems, and business decision-making.

## Dataset
The dataset used is **Amazon book reviews** from Kaggle, relabeled to classify sentiments into categories:
- Positive
- Negative
- Neutral

Each category contains **1000 samples**, with an **80/20 train-validation split**.


## Model Architecture
The improved **2-layer BiLSTM model** integrates:
- **BERT embeddings**: To provide context-aware input representations.
- **Self-attention mechanisms**: Assigning varying importance to sequence elements.
- **Pooling layers**: Reducing output dimensions and improving computational efficiency.
- **Layer normalization**: Ensuring stability and mitigating gradient issues.

![Model Diagram](path/to/your/model_diagram.png)

## Installation and Usage

### Requirements
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Running the Model
1. Clone this repository:
   ```bash
   git clone https://github.com/river-d/BiLSTM-models-for-sentiment-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BiLSTM-models-for-sentiment-analysis
   ```
3. Train the model:
   ```bash
   python train.py --config config.yaml
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py --model_path saved_model.pth
   ```

### File Structure
```
BiLSTM-models-for-sentiment-analysis/
├── data/                 # Dataset files
├── models/               # Model architecture files
├── scripts/              # Training and evaluation scripts
├── results/              # Output results and metrics
├── README.md             # Project description
└── requirements.txt      # Dependencies
```

## Citation
If you use this code or research, please cite:
```
@article{your_paper,
  title={The Comparison of Transformer and Improved BiLSTM Model for Document-level Sentiment Analysis},
  author={Tao He},
  journal={Your Journal},
  year={2024}
}
```

## Contact
For any questions or feedback, feel free to contact:
- **Tao He**: [Email](mailto:hetaoo.c@gmail.com)

