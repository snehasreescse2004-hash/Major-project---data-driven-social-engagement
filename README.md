# THE DATA-DRIVEN SOCIAL ENGAGEMENT INITIATIVE
### Evidence-Based Social Content Analytics, Predictive Virality, and Prescriptive Content Strategy
*A Production-Grade BE CSE / Data Science Major Project*

---

## 1. Executive Summary & Problem Statement

In contemporary digital media ecosystems, content creation and distribution strategies are predominantly governed by trial-and-error, subjective heuristics, and unverified intuitions. While billions of user interactions occur daily across platforms like Instagram, TikTok, and YouTube, digital creators and brand strategists often lack systematic, reproducible methods to:
1. Objectively quantify the multifactorial drivers of content virality.
2. Measure deep audience resonance and relatable feedback versus superficial praise.
3. Conduct statistically rigorous A/B comparisons on observational engagement data.
4. Prescribe data-driven content blueprints tailored to specific thematic domains.

**The Data-Driven Social Engagement Initiative** resolves these challenges by bridging classical Machine Learning (ML), Natural Language Processing (NLP), and parametric/non-parametric Statistical Hypothesis Testing into a unified decision support platform.

---

## 2. Project Architecture & System Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION LAYER                             │
│  - Flexible CSV Normalization (Synonym Alias Mapping)                     │
│  - Robust Fallback Synthetic Demo Generator (1,200+ Posts, 3,500+ Comments)│
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING & FEATURE PIPELINE                     │
│  - Deduplication & Outlier Clamping (Retention ∈ [0, 100%])               │
│  - Temporal Parsing (Hour, Day, Weekend Flag)                             │
│  - Text Normalization (Regex URL Stripping, Lowercasing, Whitespace)      │
│  - Derived Metrics: Save-Share Ratio, Comment Rate, Engagement Rates      │
│  - Transparent Formulaic Viral Score (0–100 Scale)                        │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           ▼                          ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  PREDICTIVE VIRALITY │   │   DUAL-DIMENSION NLP │   │   STATISTICAL A/B    │
│      CLASSIFIER      │   │       PIPELINE       │   │    TESTING SUITE     │
│ - StandardScaler     │   │ - TF-IDF Vectorizer  │   │ - Welch's t-test     │
│ - OneHotEncoder      │   │ - LogisticRegression │   │ - Mann-Whitney U     │
│ - RandomForest       │   │ - TextBlob Polarity  │   │ - α = 0.05 Threshold │
└──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       PRESCRIPTIVE STRATEGY ENGINE                        │
│  - Dynamic Multi-Factor Ranking (Topic, Format, Hook, Duration, Timing)   │
│  - Empirical Rationale Generator (3–5 Data-Backed Justifications)         │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    INTERACTIVE STREAMLIT DASHBOARD (10 TABS)              │
│  Executive Overview | Content Analytics | Virality Predictor | NLP Lab   │
│  A/B Testing | Trend Opportunity | Prescriptive Blueprint | Explorer      │
│  Model Evaluation | Academic Project Report                              │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
data_driven_social_engagement/
│
├── app.py                      # Main 10-page Streamlit Dashboard
├── config.py                   # Central constants, schemas, weights & themes
├── requirements.txt            # Pinned dependency manifest
├── README.md                   # Comprehensive technical documentation
├── .gitignore                  # Git ignore rules
├── .env.example                # Example configuration for external APIs
│
├── data/
│   ├── posts.csv               # Historical content metrics corpus
│   └── comments.csv            # Audience textual comments corpus
│
├── models/
│   ├── viral_model.pkl         # Trained RandomForest virality pipeline
│   ├── relatable_model.pkl     # Trained Logistic Regression relatability model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF N-gram vectorizer
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Resilient ingestion with alias mapping & fallback
│   ├── preprocessing.py        # Post/comment cleaning & date-time extraction
│   ├── feature_engineering.py  # Derived rates, ratios, & Viral Score index
│   ├── viral_model.py          # ML pipeline, feature importance & inference
│   ├── nlp_model.py            # Relatability classifier & TextBlob polarity
│   ├── ab_testing.py           # Welch's t-test & Mann-Whitney U evaluations
│   ├── trend_analysis.py       # Topic momentum & Trend Opportunity Score
│   ├── recommender.py          # Prescriptive recommendation engine
│   └── visualizations.py       # Interactive Plotly chart builders
│
├── utils/
│   ├── __init__.py
│   └── helpers.py              # Metric formatters & executive report writer
│
├── reports/
│   └── generated_insights.txt  # Auto-generated textual insights report
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py   # Tests for cleaning and bounds
    ├── test_features.py        # Tests for arithmetic & Viral Score logic
    └── test_models.py          # Tests for ML, NLP, and statistical tests
```

---

## 4. Methodological Foundations

### 4.1 The Transparent Viral Score Formulation
Rather than treating virality as an arbitrary binary label, we formulate a transparent, normalized index ($0 \text{ to } 100$):

$$\text{Viral Score} = 0.15 \tilde{L} + 0.25 \tilde{S} + 0.25 \tilde{V} + 0.20 \tilde{R} + 0.10 \tilde{C} + 0.05 \tilde{F}$$

* $\tilde{L} = \text{Normalized Likes}$ ($15\%$)
* $\tilde{S} = \text{Normalized Shares}$ ($25\%$ - Viral amplification coefficient)
* $\tilde{V} = \text{Normalized Saves}$ ($25\%$ - High-utility bookmarking index)
* $\tilde{R} = \text{Normalized Retention Rate}$ ($20\%$ - Watch completion ratio)
* $\tilde{C} = \text{Normalized Comments}$ ($10\%$ - Conversational engagement)
* $\tilde{F} = \text{Normalized Followers Gained}$ ($5\%$ - Audience conversion efficiency)

**Tiers:**
* **High Virality:** $\text{Viral Score} \ge 70$
* **Medium Virality:** $40 \le \text{Viral Score} < 70$
* **Low Virality:** $\text{Viral Score} < 40$

### 4.2 Machine Learning Virality Classifier
* **Target:** Multi-class classification into `Low`, `Medium`, and `High` virality tiers.
* **Pipeline:**
  * Numerical attributes (`likes`, `shares`, `saves`, `comments_count`, `retention_rate`, `followers_gained`, `duration`, `caption_length`, `hashtag_count`, `hour`) are standardized via `StandardScaler`.
  * Categorical attributes (`content_type`, `topic`, `hook_type`, `platform`) are one-hot encoded via `OneHotEncoder(handle_unknown='ignore')`.
  * Estimator: `RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')`.
* **Validation:** Stratified holdout test split ($80/20$), evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix heatmaps.

### 4.3 Dual-Dimension Audience NLP Pipeline
Audience feedback is analyzed across two orthogonal dimensions:
1. **Relatability Classification (`RELATABLE` vs `NEUTRAL`):**
   * Uses N-gram `TfidfVectorizer(max_features=2500, ngram_range=(1, 2))` paired with `LogisticRegression(C=1.5, class_weight='balanced')`.
   * When unlabelled comments are ingested, weak-supervision linguistic heuristics identify resonance patterns (e.g. *"literally me"*, *"so true"*, *"same here"*).
2. **Sentiment Polarity Analysis (`Positive`, `Neutral`, `Negative`):**
   * Utilizes TextBlob polarity scoring: Polarity $> 0.05 \to \text{Positive}$, $< -0.05 \to \text{Negative}$, else $\text{Neutral}$.
   * *Critical Distinction:* Negative sentiment frequently co-occurs with high relatability (e.g., discussions of workplace stress or social anxiety).

### 4.4 Statistical A/B Hypothesis Testing
Compares key strategic decisions:
* **Test A:** Hook Strategy (Visual vs Text)
* **Test B:** Content Duration (Short $<60s$ vs Long $\ge60s$)
* **Test C:** Publishing Window (Morning $06:00-12:00$ vs Evening $17:00-23:00$)
* **Custom Tests:** Dynamic cohort comparisons on any metric.
* **Methods:**
  * **Welch's t-test:** Accounts for potential unequal variances between observational cohorts.
  * **Mann-Whitney U Test:** Non-parametric evaluation robust to extreme non-normality and skewness.
  * **Significance Level:** $\alpha = 0.05$.
  * *Observational Caveat:* Observational associations do not satisfy randomized controlled trial (RCT) requirements and should not be claimed as absolute causal proofs.

### 4.5 Trend Opportunity Analysis
Topic momentum is quantified via a composite Trend Opportunity Index:

$$\text{Trend Score} = 0.40 \cdot \tilde{G} + 0.30 \cdot \tilde{E} + 0.30 \cdot \tilde{F}$$

* $\tilde{G} = \text{Normalized Recent vs Earlier Chronological Growth Rate}$
* $\tilde{E} = \text{Normalized Mean Engagement}$
* $\tilde{F} = \text{Normalized Post Frequency}$

---

## 5. Technology Stack

* **Language:** Python 3.10+ (Tested & verified on Python 3.14)
* **Core Analytics & Data Structures:** `pandas`, `numpy`, `scipy`
* **Machine Learning & Pipeline:** `scikit-learn`, `joblib`
* **Natural Language Processing:** `nltk`, `textblob`
* **Data Visualization:** `plotly`, `matplotlib`, `seaborn`
* **Web Dashboard Application:** `streamlit`

---

## 6. Installation & Execution Instructions

### Step 1: Clone or Navigate to Project Directory
```powershell
cd "c:\Users\Admin\Desktop\Unlox major project\data_driven_social_engagement"
```

### Step 2: Install Verified Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Automated Unit Tests
```powershell
python -m unittest discover tests
```
*Expected Output:*
```text
.........
----------------------------------------------------------------------
Ran 9 tests in 0.969s

OK
```

### Step 4: Launch the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
*The interactive dashboard will automatically open at `http://localhost:8501`.*

*(Optional) You can also run directly from the workspace root folder:*
```powershell
cd "c:\Users\Admin\Desktop\Unlox major project"
python -m streamlit run app.py
```

---

## 7. How to Demonstrate the Project (Viva / Evaluation Walkthrough)

1. **Executive Overview (Tab 1):**
   * Demonstrate the 6 responsive KPI metric cards (Total Posts, Total Engagement, Viral Score, etc.).
   * Highlight the dynamic **Key Analytical Insights** automatically derived from the active data slice.
   * Filter by platform (e.g., select only Instagram) or topic and show instant metric recalculation.
2. **Content Analytics (Tab 2):**
   * Inspect the sorted top-performing posts table.
   * Demonstrate the **Save-to-Share Ratio** chart illustrating content utility vs social currency.
   * Click **Download Filtered Content Analytics CSV** to show export capabilities.
3. **Virality Prediction (Tab 3):**
   * Present the mathematical formulation expander.
   * Use the interactive input form, adjust sliders (Likes, Shares, Saves, Retention), and click **⚡ PREDICT VIRALITY**.
   * Show the returned virality category (`HIGH`, `MEDIUM`, or `LOW`), confidence level, and class probabilities.
4. **Audience NLP (Tab 4):**
   * Explain the difference between Sentiment Polarity and Relatability.
   * Type in custom audience text (e.g. *"This describes my whole life, so accurate!"*), click **🔍 ANALYZE COMMENT**, and display the dual prediction.
5. **A/B Testing (Tab 5):**
   * Walk through the Hook Type (Visual vs Text) and Posting Time hypothesis tests.
   * Explain the p-value and point out the methodological disclaimer on observational association.
6. **Trend Analysis & Recommendations (Tabs 6 & 7):**
   * Review the Trend Opportunity ranking.
   * Present the **Optimal Content Strategy Blueprint** card with its 5 empirical data-backed justifications.
7. **Model Performance (Tab 9):**
   * Show the Confusion Matrix heatmaps and Gini feature importances for RandomForest and TF-IDF models.

---

## 8. Honest Project Limitations

1. **Observational Association vs Causality:** Observational A/B comparisons do not meet randomized controlled trial (RCT) criteria and cannot prove definitive causation.
2. **Synthetic Data Transparency:** When live platform credentials or user CSVs are absent, the application utilizes a realistic synthetic demo corpus. It is clearly labeled as synthetic in the UI.
3. **Relatability Nuance:** Natural language relatability is subtle and relies on weak-supervision heuristics in the absence of human-annotated ground truth.
4. **Platform Index Formulation:** The Viral Score is a project-defined analytical index rather than a universal physical constant.

---

## 9. Future Roadmap

* Direct OAuth2 integration with Meta Graph API (Instagram) and YouTube Data API v3.
* Transformer-based LLM sentiment and relatability classification using fine-tuned RoBERTa / DeBERTa.
* Multilingual and multi-modal support (analyzing video thumbnails and audio transcriptions).
* Time-series forecasting for creator follower growth trajectories using Prophet / NeuralProphet.
