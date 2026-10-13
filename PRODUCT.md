# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Food tech ML engineers, NLP practitioners, and evaluators benchmarking NLP models on Thai text, inspecting real-time rating consistency, and debugging text-rating discrepancies in food review datasets.

## Product Purpose
Kin Nai Dee (กินไหนดี) is an NLP and Deep Learning consistency inspection and benchmarking web application designed to detect contradictions between Thai restaurant review texts and their assigned star ratings (1–5 stars), while providing real-time inference latency and engineering trade-off evaluations between Deep Learning (BiLSTM + thai2vec) and Classical Machine Learning (LinearSVC + TF-IDF).

## Positioning
Unlike generic sentiment analysis demos, Kin Nai Dee focuses specifically on fine-grained 5-class consistency verification for Thai text, detecting nuanced edge cases such as sarcasm, noise, mixed sentiment (praising food while criticizing service), and providing side-by-side engineering benchmarks (latency, memory footprint, throughput, and class-imbalanced recall).

## Operating Context
- Web application interface running locally on Flask (Python backend, HTML/CSS/JS frontend).
- Live inspection workflows for custom user-submitted Thai reviews or curated edge-case presets.
- Experimental analysis and model defense workflows referencing offline test benchmarks, confusion matrices, and error analysis logs.

## Capabilities and Constraints
- Dual-model inference: Proposed BiLSTM (PyTorch with thai2vec embeddings) and Baseline LinearSVC (scikit-learn with TF-IDF features).
- Real-time latency measurement in milliseconds per inference request.
- Classification into 1–5 stars with confidence/probability distributions across all 5 classes.
- Contradiction alerting with severity indicators (Consistent, Minor Discrepancy, Severe Contradiction).
- Important token highlighting (positive/negative sentiment indicators using PyThaiNLP tokenization).
- Engineering benchmark comparison (Macro F1, 1-Star Recall, Latency, Throughput, Model Size).
- Pre-computed visual evaluation assets: Confusion matrices, F1 vs Latency trade-offs, and class imbalance distributions.

## Brand Commitments
- Name: Kin Nai Dee (กินไหนดี) — Rating & Latency Inspector.
- Aesthetic: Modern dark-mode engineering dashboard with ambient glow accents, clean typography (Plus Jakarta Sans + Prompt for Thai), and high-contrast diagnostic indicators.

## Evidence on Hand
- Trained model checkpoints for BiLSTM and LinearSVC.
- Empirical evaluation outputs in `outputs/` and `imbalance_compare/` containing experiment metrics, confusion matrices, and benchmark summaries.
- Curated edge-case presets in the web app (e.g. service complaints with 1 star, sarcastic reviews, accidental 1-star ratings).

## Product Principles
- Engineering-First Transparency: Present latency, confidence distribution, and model limitations side-by-side with predictions.
- No Silent Assumptions: Clearly distinguish between text sentiment prediction and user-submitted rating to spotlight genuine contradictions.
- Real-Time Actionability: Provide instantaneous feedback and diagnostic token highlights for rapid inspection and debugging.
- Empirical Rigor: Back up live predictions with documented benchmarks, trade-off rules, and error analyses.
