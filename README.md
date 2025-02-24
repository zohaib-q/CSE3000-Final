# CSE3000-Final
CSE 3000 - AI Content Moderation System
Objective
Build a simple AI model to classify text as appropriate or inappropriate (e.g., toxic, offensive,
or spam) using a real dataset. Analyze its performance, discuss ethical concerns, and propose
improvements for fairness and transparency.

Assignment Details

Part 1: Build a Content Moderation Model

  • Dataset: Use a publicly available dataset such as:
  
    – Jigsaw Toxic Comment Dataset (Kaggle)
    
    – Twitter Hate Speech Dataset (available on GitHub or other repositories)
    
    – https://hatespeechdata.com/
    
    – https://github.com/aymeam/Datasets-for-Hate-Speech-Detection
    
  • Text Classification:
  
  1. Preprocess the dataset:
    
      – Clean text (e.g., remove special characters, stopwords).
      
      – Tokenize and convert text to numerical features using libraries like CountVectorizer
        or TfidfVectorizer.
        
  2. Train a simple machine learning model (e.g., Logistic Regression or Naive Bayes)
        to classify text as:
        
      – Toxic
      
      – Non-toxic
      
      – Or any other provided labels.
      
  3. Evaluate Performance:
    
      – Use metrics like accuracy, precision, recall, and F1-score.
      
      – Split data into training and test sets (e.g., 80/20).

Part 2: Ethical Analysis

  • Analyze Model Bias:
  
   – Check if the model disproportionately flags text from certain groups as toxic (e.g.,
      based on gender, race, or context).
      
   – Use the dataset’s metadata if available (e.g., demographic labels).
    
  • Propose Improvements:
  
   – Suggest ways to reduce bias (e.g., balanced training data, threshold adjustments).
    
   – Reflect on how false positives/negatives could impact users (e.g., silencing marginalized voices or allowing harmful content).

Deliverables

  • Code Submission:
  
   – A Jupyter Notebook or Python script implementing the model and evaluations.
    
   – Include well-documented code and instructions.
    
  • Presentation (5-10 minutes):
  
   – Visualizations of performance metrics (e.g., confusion matrix, precision-recall
      curve).
      
   – Discussion of bias, limitations, and ethical concerns.
    
  • Technical Report (2-3 pages):
  
   – Summarize methodology, results, and ethical reflections.

Example Workflow

  • Preprocess Data: Clean, tokenize, and convert text to numerical features.
  
  • Train and Evaluate Model: Train a model and evaluate using test data.
  
  • Visualize Results: Plot a confusion matrix or precision-recall curve using matplotlib
    or seaborn.
    
    
Ethical Reflection Prompts

   • How might your model’s false positives (flagging non-toxic comments) affect user experience?
    
   • If toxic comments from certain groups are more frequently flagged, what steps could
      mitigate this bias?
      
   • Should platforms prioritize accuracy over inclusivity, or vice versa?
    

Grading Criteria

  • Presentation and Report (40%): Clarity of methodology and ethical reflections.
  
  • Bias Analysis (30%): Depth and quality of bias evaluation.
  
  • Model Performance (20%): Correct implementation and evaluation of the content
    moderation model.
    
  • Creativity (10%): Innovative ideas for improving model fairness.
