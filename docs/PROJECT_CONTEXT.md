# Project Context: Machine Listener System

## 1. Project Overview & Environment

- **Institution:** Cairo University, Faculty of Engineering.
- **Department:** Computer Engineering Department.
- **Course:** Pattern Recognition and Neural Networks.
- **Term:** Spring 2026.
- **Project Title:** Machine Fault Recognition / Machine Listener System.

## 2. Core Objective

The primary goal is to design and implement a complete machine learning pipeline capable of classifying audio files into one of 6 distinct classes representing the operational state of different machines.

**Classification Labels:**

- `0`: Machine 1, Normal
- `1`: Machine 1, Abnormal
- `2`: Machine 2, Normal
- `3`: Machine 2, Abnormal
- `4`: Machine 3, Normal
- `5`: Machine 3, Abnormal

## 3. System Specifications & Pipeline

The developed system must be robust against variations and consist of the following integrated modules:

- **Preprocessing Module:** Must handle background noise, volume variations, silence removal, and different microphone qualities.
- **Feature Extraction/Selection Module:** To extract meaningful data from the raw audio signals.
- **Model Selection and Training Module:** Training the classifier.
- **Performance Analysis Module:** Evaluating the model's efficacy.

## 4. Constraints & Technical Rules

- **Dataset Split:** The provided dataset must be divided into Training, Validation (for hyperparameter tuning), and Test sets.
- **Model Limitations:** Both classical ML and Deep Learning architectures are permitted; however, models are capped at a maximum of 2 Billion parameters running at once.
- **Language:** Any programming language is allowed, but execution speed is a grading factor.

## 5. Grading Criteria

This is a competition-based project ranked on the following distribution:

- **Report & Discussion (20%):** Evaluates report quality and the team's understanding during the final discussion.
- **Project Performance (80%):**
  - **Accuracy:** 65%
  - **Running Time:** 15%

## 6. Evaluation format (I/O Specifications)

The model will be evaluated against a hidden test set in a specific format.

**Input Data:**

- A directory named `data/` containing sequentially named `.wav` files (e.g., `1.wav`, `2.wav`, `...`, `100.wav`).
- Files must be processed in strictly increasing numerical order.

**Output Deliverables (Generated at Runtime):**
The execution must generate two text files, formatted precisely:

1.  **`results.txt`:** Contains the predicted class integer (0-5) for each test case, one per line, strictly matching the input file order. No extra text or test numbers.
2.  **`time.txt`:** Contains the execution time in seconds for each iteration (including preprocessing but excluding I/O overhead), rounded to three decimal places, one per line.

## 7. Submission Deliverables

- **Report:** A PDF following an Overleaf template covering Abstract, Intro, System Overview, Experimental Setup, Results, Conclusion, and Workload Distribution.
- **Source Code:** A zipped folder (`code_[team number].zip`) containing a `readme.md`, `requirements.txt`, and an execution script named `infer.py` (which takes the `data` directory path as an argument).
- **Model File:** The trained weights (Pickle format recommended), correctly pathed to load during execution.
- **Environment:** A Dockerfile to replicate the exact execution environment.
- **Bonus:** Cloud deployment (e.g., Hugging Face).
