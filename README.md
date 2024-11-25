# Brazilian Financial News Sentiment Analysis

This repository contains the implementation of the study "The Relationship Between Financial News in the US and the Brazilian Real Estate Investment Trust Market: A Sentiment Analysis Perspective". This study investigates the relationship between the sentiment of US financial news and the performance of the Brazilian Real Estate Investment Trust (FII) market, as measured by the IFIX index. Using sentiment analysis techniques and time series modeling, it was examined whether there is a significant relation between these factors. The VADER tool was used for sentiment analysis of news headlines and snippets obtained from Google News, and Vector Autoregressive (VAR) models were applied to analyze the relationship between news sentiment and IFIX values. From the obtained models, Granger causality tests were conducted, and impulse response functions (IRF) of the variables were analyzed to verify the existence of a significant relation between the variables, but the results indicated no significant relation. These findings contrast with previous studies that found relations between news sentiment and other Brazilian financial indicators. Therefore, possible limitations of this study were acknowledged, including the chosen economic indicator (IFIX) and the nature of the news dataset used.

## Project Overview

The study involves:
1. Creating a custom dataset of financial news about Brazil from reliable sources
2. Performing sentiment analysis on news titles and snippets using VADER
3. Analyzing the relationship between news sentiment and IFIX using Vector Auto Regression (VAR)

## Dataset Creation

The dataset was built using:
- **Source**: Google News API via HasData platform
- **Time Period**: December 2021 to August 2024
- **News Sources**: Bloomberg and Reuters
- **Total Articles**: 2,328 news articles
- **Content**: Financial news specifically related to Brazil
- **Language**: English
- **Target Audience**: US readers

### Data Collection Process

The news collection process involved:
- Custom Python scripts for API calls
- Filtering for Brazil-related content
- Quality checks for source reliability
- Removal of duplicates and irrelevant content
- Validation of Brazilian context through keyword matching



## Methodology

### News Sentiment Analysis
- **Tool**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Analysis Components**:
  - Title sentiment
  - Snippet sentiment
  - Average sentiment (combined score)

### Time Series Analysis
- **Model**: Vector Auto Regression (VAR)
- **Features**:
  - Stationarity testing using Dickey-Fuller test
  - Series transformation through differentiation
  - Optimal lag selection using BIC and AIC
  - Granger Causality testing
  - Impulse Response Function analysis

## Usage

1. **Data Collection**:
```python
a_news_scrapper.py
ifix_processor_bulk.py
```

2. **Data Processing**:
```python
b_news_processing.py
```

3. **Sentiment Analysis**:
```python
c_news_sentiment_analyzing.py
```

4. **Time Series Analysis**:
```python
d_sentiment_ifix_preparation.py
e_sentiment_ifix_stationarity_var.py
f_sentiment_ifix_var_subset.py
g_sentiment_ifix_var_monthly.py
```
