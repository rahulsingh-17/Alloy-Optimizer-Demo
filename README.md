# Alloy Composition Optimization (Copper-Based Alloys)

This project focuses on predicting the optimal alloy composition for **copper-based alloys** based on user-defined mechanical and electrical property requirements.

## Overview

The system uses machine learning models to recommend alloy compositions that best match desired target properties, including:

- Electrical Conductivity  
- Hardness  
- Tensile Strength  
- Yield Strength  

## Models Used

Two regression models are implemented and compared:

- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  

These models are trained on a dataset of copper alloy compositions and their corresponding material properties.

## Input Methods

The system supports two types of user inputs:

### 1. Keyword-Based NLP Input

- Users can enter desired properties in natural language  
- Example:  
  "I need an alloy with high conductivity and moderate strength"  
- Keyword matching is used to map terms like "high", "low", and "moderate" to numerical ranges  

### 2. Numeric Range Input

- Users can directly specify minimum and maximum values for each property  

Example:
- Conductivity: 80–95 %IACS  
- Tensile Strength: 300–500 MPa  

## Workflow

1. User provides desired property values (via keyword input or numeric input)  
2. Input is processed and converted into structured format  
3. Trained ML models predict the optimal alloy composition  
4. Output includes recommended composition and expected properties  

## Dataset

- Based on copper alloy compositions  
- Includes:
  - Elemental composition (percentage of elements)  
  - Measured physical and mechanical properties  

## Goals

- Assist in materials design and optimization  
- Reduce trial-and-error in alloy development  
- Provide a data-driven approach for engineering decisions  

## Future Improvements

- Add more alloy systems (e.g., aluminum, steel)  
- Improve keyword-based NLP with advanced language models  
- Integrate optimization algorithms (e.g., genetic algorithms)  
- Improve UI/UX of the web application  

## Requirements

- Python 3.x  
- scikit-learn  
- pandas  
- numpy  
