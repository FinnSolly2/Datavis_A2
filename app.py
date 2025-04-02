import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import io
import pickle
import base64

# Set page config
st.set_page_config(page_title="ML Model Trainer", layout="wide")

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = None
if 'is_classification' not in st.session_state:
    st.session_state.is_classification = False
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'roc_data' not in st.session_state:
    st.session_state.roc_data = None

# Main title
st.title("Machine Learning Model Trainer")

# Create sidebar
st.sidebar.title("Model Configuration")

# Dataset selection
st.sidebar.header("1. Select Dataset")

# Upload custom dataset
uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])

dataset_option = st.sidebar.radio(
    "Choose a dataset source:",
    ["Sample Datasets", "Upload Your Own"],
    index=0
)

if dataset_option == "Sample Datasets":
    sample_datasets = {
        "Tips": sns.load_dataset("tips"),
        "Titanic": sns.load_dataset("titanic"),
        "Diamonds": sns.load_dataset("diamonds"),
        "Iris": sns.load_dataset("iris"),
        "Penguins": sns.load_dataset("penguins"),
        "Planets": sns.load_dataset("planets"),
        "Cars": sns.load_dataset("mpg")
    }
    
    dataset_name = st.sidebar.selectbox("Select a dataset", list(sample_datasets.keys()))
    
    if dataset_name:
        # Clear model-related session state when dataset changes
        if st.session_state.dataset is not None and not st.session_state.dataset.equals(sample_datasets[dataset_name]):
            st.session_state.trained_model = None
            st.session_state.feature_importance = None
            st.session_state.metrics = {}
            st.session_state.predictions = None
            st.session_state.y_test = None
            st.session_state.X_test = None
            st.session_state.roc_data = None
        
        st.session_state.dataset = sample_datasets[dataset_name]
        st.session_state.dataset_type = "Sample"
elif dataset_option == "Upload Your Own" and uploaded_file is not None:
    try:
        # Clear model-related session state when a new dataset is uploaded
        if st.session_state.dataset_type != "Custom" or st.session_state.dataset is None:
            st.session_state.trained_model = None
            st.session_state.feature_importance = None
            st.session_state.metrics = {}
            st.session_state.predictions = None
            st.session_state.y_test = None
            st.session_state.X_test = None
            st.session_state.roc_data = None
            
        st.session_state.dataset = pd.read_csv(uploaded_file)
        st.session_state.dataset_type = "Custom"
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")

# Show dataset preview if available
if st.session_state.dataset is not None:
    with st.expander("Dataset Preview", expanded=True):
        # Display dataset preview
        st.write(f"Dataset shape: {st.session_state.dataset.shape}")
        
        # Convert to string to avoid Arrow conversion issues
        preview_df = st.session_state.dataset.head().copy()
        for col in preview_df.columns:
            if preview_df[col].dtype.name not in ['int64', 'float64']:
                preview_df[col] = preview_df[col].astype(str)
                
        st.dataframe(preview_df)
        
        # Display dataset info in a dataframe
        df_info = pd.DataFrame({
            'Column': st.session_state.dataset.columns,
            'Type': st.session_state.dataset.dtypes.astype(str),  # Convert dtype objects to strings
            'Non-Null Count': st.session_state.dataset.count(),
            'Null Count': st.session_state.dataset.isna().sum(),
            'Null %': (st.session_state.dataset.isna().sum() / len(st.session_state.dataset) * 100).round(2),
            'Unique Values': [st.session_state.dataset[col].nunique() for col in st.session_state.dataset.columns]
        })
        st.write("Dataset Information:")
        st.dataframe(df_info.astype(str), use_container_width=True)  # Convert all values to strings
        
        # Display dataset statistics
        st.write("Dataset Statistics:")
        try:
            # Try to create a statistics dataframe safely
            stats_df = st.session_state.dataset.describe().reset_index()
            stats_df = stats_df.astype(str)  # Convert all to strings to avoid Arrow issues
            st.dataframe(stats_df)
        except Exception as e:
            st.warning(f"Could not generate all statistics: {e}")
            # Fallback to individual statistics
            numeric_cols = st.session_state.dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                st.write(f"Numeric columns: {', '.join(numeric_cols)}")
                for col in numeric_cols[:5]:  # Show stats for first 5 numeric columns
                    try:
                        st.write(f"Statistics for {col}:")
                        col_stats = pd.DataFrame({
                            'Statistic': ['mean', 'std', 'min', '25%', '50%', '75%', 'max'],
                            'Value': [
                                str(st.session_state.dataset[col].mean()),
                                str(st.session_state.dataset[col].std()),
                                str(st.session_state.dataset[col].min()),
                                str(st.session_state.dataset[col].quantile(0.25)),
                                str(st.session_state.dataset[col].quantile(0.5)),
                                str(st.session_state.dataset[col].quantile(0.75)),
                                str(st.session_state.dataset[col].max())
                            ]
                        })
                        st.dataframe(col_stats)
                    except:
                        pass

# Feature selection
if st.session_state.dataset is not None:
    st.sidebar.header("2. Select Features and Target")
    
    # Get numerical and categorical columns
    numerical_cols = st.session_state.dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = st.session_state.dataset.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Target variable selection
    all_cols = numerical_cols + categorical_cols
    
    # Add information about columns with missing values
    cols_with_missing = st.session_state.dataset.columns[st.session_state.dataset.isna().any()].tolist()
    if cols_with_missing:
        st.sidebar.warning(f"⚠️ The following columns contain missing values which may cause issues if selected as target: {', '.join(cols_with_missing)}")
    
    target_variable = st.sidebar.selectbox("Select target variable", all_cols)
    
    if target_variable in numerical_cols:
        st.session_state.is_classification = st.sidebar.checkbox("Treat as classification task", 
                                                                help="Enable for numerical target variables that represent categories (e.g., 0/1 values). Disable for true regression tasks.")
    else:
        st.session_state.is_classification = True
        st.sidebar.info("Target variable is categorical - classification task selected automatically.")
    
    # Feature selection
    available_features = [col for col in all_cols if col != target_variable]
    
    with st.sidebar.expander("Select numerical features", expanded=True):
        numerical_features = [col for col in numerical_cols if col != target_variable]
        selected_numerical = st.multiselect(
            "Choose numerical features",
            numerical_features,
            default=numerical_features
        )
    
    with st.sidebar.expander("Select categorical features", expanded=True):
        categorical_features = [col for col in categorical_cols if col != target_variable]
        selected_categorical = st.multiselect(
            "Choose categorical features",
            categorical_features,
            default=categorical_features
        )
    
    # Combine selected features
    selected_features = selected_numerical + selected_categorical
    
    # Model selection
    st.sidebar.header("3. Select and Configure Model")
    
    # Model type selection based on task
    if st.session_state.is_classification:
        model_options = ["Logistic Regression", "Random Forest Classifier"]
    else:
        model_options = ["Linear Regression", "Random Forest Regressor"]
    
    model_type = st.sidebar.selectbox("Select model type", model_options)
    st.session_state.model_type = model_type
    
    # Model parameters
    with st.sidebar.form(key="model_params_form"):
        st.header("Model Parameters")
        
        # Data handling parameters
        st.subheader("Data Preparation")
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 100, 42)
        
        # Missing value handling
        st.subheader("Missing Value Handling")
        numeric_impute_strategy = st.selectbox(
            "Numeric imputation strategy", 
            options=["mean", "median", "most_frequent", "constant"],
            index=0,
            help="Strategy to fill missing values in numeric columns"
        )
        
        if numeric_impute_strategy == "constant":
            numeric_fill_value = st.number_input("Numeric fill value", value=0)
        else:
            numeric_fill_value = None
            
        categorical_impute_strategy = st.selectbox(
            "Categorical imputation strategy", 
            options=["most_frequent", "constant"],
            index=0,
            help="Strategy to fill missing values in categorical columns"
        )
        
        if categorical_impute_strategy == "constant":
            categorical_fill_value = st.text_input("Categorical fill value", value="missing")
        else:
            categorical_fill_value = None
        
        # Model-specific parameters
        st.subheader("Model Parameters")
        if model_type in ["Random Forest Regressor", "Random Forest Classifier"]:
            n_estimators = st.number_input("Number of estimators", 10, 500, 100, 10)
            max_depth = st.number_input("Maximum depth", 1, 100, 10)
            min_samples_split = st.number_input("Minimum samples split", 2, 20, 2)
        
        if model_type == "Logistic Regression":
            C = st.number_input("Regularization parameter (C)", 0.01, 10.0, 1.0, 0.1)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"])
            max_iter = st.number_input("Maximum iterations", 100, 1000, 100, 100)
        
        # Form submit button
        fit_button = st.form_submit_button("Fit Model")

    # Function to train model
    def train_model():
        st.session_state.feature_names = selected_features.copy()
        X = st.session_state.dataset[selected_features]
        y = st.session_state.dataset[target_variable]
        
        # Check for missing values in the target variable
        if y.isna().any():
            st.error(f"Error: The target variable '{target_variable}' contains missing values (NaN). Please select a different target variable or preprocess your data to handle missing values.")
            return False
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Save test data for evaluation
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy=numeric_impute_strategy,
                fill_value=numeric_fill_value if numeric_impute_strategy == 'constant' else None
            )),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy=categorical_impute_strategy,
                fill_value=categorical_fill_value if categorical_impute_strategy == 'constant' else None
            )),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, selected_numerical),
                ('cat', categorical_transformer, selected_categorical)
            ]
        )
        
        # Create model
        if model_type == "Linear Regression":
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
        elif model_type == "Random Forest Regressor":
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state
                ))
            ])
        elif model_type == "Logistic Regression":
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=random_state
                ))
            ])
        elif model_type == "Random Forest Classifier":
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state
                ))
            ])
        
        # Train model
        model.fit(X_train, y_train)
        st.session_state.trained_model = model
        
        # Make predictions
        st.session_state.predictions = model.predict(X_test)
        
        # Calculate metrics
        if st.session_state.is_classification:
            accuracy = accuracy_score(y_test, st.session_state.predictions)
            precision = precision_score(y_test, st.session_state.predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, st.session_state.predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, st.session_state.predictions, average='weighted', zero_division=0)
            
            st.session_state.metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
            
            # For ROC curve (binary classification)
            try:
                # Check if data is suitable for ROC curve (binary classification with predict_proba)
                y_unique = np.unique(y)
                is_binary = len(y_unique) == 2
                has_predict_proba = hasattr(model[-1], "predict_proba")
                
                if is_binary and has_predict_proba:
                    # Handle non-numeric labels for ROC curve
                    if pd.api.types.is_numeric_dtype(y_test):
                        # If already numeric (0,1), we can use directly
                        y_prob = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                    else:
                        # If categorical (e.g., 'male'/'female'), we need to encode and specify pos_label
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_test_encoded = le.fit_transform(y_test)
                        # Get second class as positive class
                        pos_label = 1
                        pos_class = le.inverse_transform([pos_label])[0]
                        
                        # Find index of positive class in predict_proba output
                        classes = model[-1].classes_
                        pos_idx = np.where(classes == pos_class)[0][0]
                        
                        y_prob = model.predict_proba(X_test)[:, pos_idx]
                        fpr, tpr, _ = roc_curve(y_test_encoded, y_prob, pos_label=pos_label)
                    
                    roc_auc = auc(fpr, tpr)
                    st.session_state.metrics["ROC AUC"] = roc_auc
                    st.session_state.roc_data = (fpr, tpr, roc_auc)
            except Exception as e:
                st.session_state.roc_data = None
                print(f"Could not generate ROC curve: {e}")
        else:
            mse = mean_squared_error(y_test, st.session_state.predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, st.session_state.predictions)
            
            st.session_state.metrics = {
                "Mean Squared Error": mse,
                "Root Mean Squared Error": rmse,
                "R² Score": r2
            }
        
        # Extract feature importance
        if model_type in ["Random Forest Regressor", "Random Forest Classifier"]:
            try:
                # Get feature names from preprocessor
                feature_names_out = []
                
                # Handle numerical features (no transformation of names)
                if selected_numerical:
                    feature_names_out.extend(selected_numerical)
                
                # Handle categorical features (need to get the one-hot encoded names)
                if selected_categorical:
                    # Check if the pipeline has been fit and has the transformers_ attribute
                    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                        preprocessor = model.named_steps['preprocessor']
                        if hasattr(preprocessor, 'transformers_'):
                            for name, transformer, cols in preprocessor.transformers_:
                                if name == 'cat' and hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                                    onehot = transformer.named_steps['onehot']
                                    if hasattr(onehot, 'get_feature_names_out'):
                                        cat_features = onehot.get_feature_names_out(selected_categorical)
                                        feature_names_out.extend(cat_features)
                                    else:
                                        # Fallback: just add the categorical feature names
                                        feature_names_out.extend([f"{col}_encoded" for col in selected_categorical])
                
                # Get feature importances from the model
                if model_type == "Random Forest Regressor":
                    importances = model.named_steps['regressor'].feature_importances_
                else:
                    importances = model.named_steps['classifier'].feature_importances_
                
                # If feature names don't match, create generic names
                if len(importances) != len(feature_names_out):
                    feature_names_out = [f"Feature {i}" for i in range(len(importances))]
                
                # Match importances with feature names
                feature_importance_data = sorted(
                    zip(feature_names_out, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                st.session_state.feature_importance = feature_importance_data
            except Exception as e:
                st.warning(f"Could not extract feature importance: {e}")
                st.session_state.feature_importance = None
    
    # Train model if button is clicked
    if fit_button:
        with st.spinner("Training model..."):
            model_trained = train_model()
        if model_trained is not False:  # Only show success if there was no error
            st.success("Model trained successfully!")

# Display model results
if st.session_state.trained_model is not None:
    st.header("Model Results")
    
    # Display metrics
    st.subheader("Model Performance Metrics")
    try:
        metrics_df = pd.DataFrame({
            "Metric": list(st.session_state.metrics.keys()),
            "Value": [str(val) for val in st.session_state.metrics.values()]  # Convert to strings
        })
        st.table(metrics_df)  # Use table instead of dataframe for better display
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")
        # Fallback to simple text display
        for metric, value in st.session_state.metrics.items():
            st.write(f"{metric}: {value}")
    
    # Create columns for plots
    col1, col2 = st.columns(2)
    
    # Create visualizations
    with col1:
        if st.session_state.is_classification:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            st.pyplot(fig)
            
            # ROC Curve (for binary classification)
            if 'ROC AUC' in st.session_state.metrics and hasattr(st.session_state, 'roc_data'):
                st.subheader("ROC Curve")
                fpr, tpr, roc_auc = st.session_state.roc_data
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                # Display a message if this is classification but not binary
                if st.session_state.is_classification:
                    st.info("ROC curve is only available for binary classification problems with predict_proba support.")
        else:
            st.subheader("Residual Plot")
            try:
                # Check if values are numeric before calculating residuals
                if pd.api.types.is_numeric_dtype(st.session_state.y_test) and pd.api.types.is_numeric_dtype(st.session_state.predictions):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    residuals = st.session_state.y_test - st.session_state.predictions
                    ax.scatter(st.session_state.predictions, residuals)
                    ax.axhline(y=0, color='r', linestyle='-')
                    ax.set_xlabel('Predicted values')
                    ax.set_ylabel('Residuals')
                    ax.set_title('Residual Plot')
                    st.pyplot(fig)
                else:
                    st.warning("Cannot create residual plot: target or predictions are not numeric data types.")
                    st.info("Consider selecting a numeric target variable for regression tasks.")
            except Exception as e:
                st.error(f"Error creating residual plot: {e}")
                st.info("This may be because the target or predictions contain non-numeric values.")
            
            st.subheader("Residual Distribution")
            try:
                # Check if values are numeric before calculating residuals
                if pd.api.types.is_numeric_dtype(st.session_state.y_test) and pd.api.types.is_numeric_dtype(st.session_state.predictions):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    residuals = st.session_state.y_test - st.session_state.predictions
                    sns.histplot(residuals, kde=True, ax=ax)
                    ax.axvline(x=0, color='r', linestyle='-')
                    ax.set_xlabel('Residual value')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Residual Distribution')
                    st.pyplot(fig)
                else:
                    st.warning("Cannot create residual distribution: target or predictions are not numeric data types.")
            except Exception as e:
                st.error(f"Error creating residual distribution: {e}")

    
    with col2:
        # Feature importance plot (for Random Forest models)
        if st.session_state.feature_importance is not None:
            st.subheader("Feature Importance")
            
            try:
                # Sort feature importance
                sorted_importance = sorted(st.session_state.feature_importance, key=lambda x: x[1])
                
                # Get top 15 features (or all if less than 15)
                num_features = min(15, len(sorted_importance))
                top_features = sorted_importance[-num_features:]
                
                features, importance = zip(*top_features)
                
                fig, ax = plt.subplots(figsize=(10, 12))
                ax.barh(features, importance)
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                st.pyplot(fig)
                
                # Also show as a table
                importance_df = pd.DataFrame(st.session_state.feature_importance, columns=['Feature', 'Importance'])
                importance_df['Importance'] = importance_df['Importance'].round(4)
                importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
                
                with st.expander("Feature Importance Table"):
                    st.dataframe(importance_df.astype(str))
            except Exception as e:
                st.error(f"Error displaying feature importance: {e}")
        else:
            st.info("Feature importance is only available for Random Forest models.")
        
        # Actual vs Predicted plot (for regression)
        if not st.session_state.is_classification:
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(st.session_state.y_test, st.session_state.predictions, alpha=0.5)
            ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()], 
                    [st.session_state.y_test.min(), st.session_state.y_test.max()], 
                    'r--')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted')
            st.pyplot(fig)
    
    # Model export
    if st.session_state.trained_model is not None:
        st.header("Export Model")
        
        model_bytes = pickle.dumps(st.session_state.trained_model)
        b64 = base64.b64encode(model_bytes).decode()
        href = f'<a href="data:file/pkl;base64,{b64}" download="trained_model.pkl">Download Trained Model</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.caption("Note: To load this model in your application, use the following code:")
        st.code("""
import pickle
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
        """)
