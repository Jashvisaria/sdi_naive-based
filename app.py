import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

def main():
    st.title("Naive Bayes Classifier App")

    # 1. Load the dataset
    # "input=datasets" or upload
    data_source = st.radio("Select Data Source", ["Iris Dataset", "Upload CSV"])
    
    df = None

    if data_source == "Iris Dataset":
        iris = load_iris()
        # Create DataFrame for easier UI handling
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        # Map target indices to names for better readability
        df['species'] = [iris.target_names[i] for i in iris.target]
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    
    if df is not None:
        st.write("### Data Preview")
        st.dataframe(df.head())

        # Select Task Type
        task_type = st.radio("Select Task Type", ["Classification", "Regression"])

        # 2. Target Variable
        # "what is ur target variable"
        if task_type == "Regression":
            target_options = df.select_dtypes(include=['number']).columns.tolist()
        else:
            # Filter for Classification: Categorical or low-cardinality numeric columns
            target_options = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() < 20]
            
        target_col = st.selectbox("Select Target Variable", target_options, index=len(target_options)-1 if target_options else 0)

        # 3. Feature Selection
        # "what is ur feture(what independent variables)"
        # GaussianNB requires numeric features, so we filter for numbers only.
        feature_cols = [c for c in df.select_dtypes(include=['number']).columns if c != target_col]
        st.info("Only numeric columns are shown below because the selected models require numerical input.")
        selected_features = st.multiselect("Select Features (Independent Variables)", feature_cols, default=feature_cols)

        # 4. Train Test Split
        # "train test speed(35%)"
        test_size = st.slider("Train/Test Split Size", 0.05, 0.95, 0.35)

        # 5. Evaluate Model Button
        if st.button("Evaluate Model"):
            if not selected_features:
                st.error("Please select at least one feature.")
            else:
                X = df[selected_features]
                y = df[target_col]
                
                if task_type == "Classification":
                    # Validation: Check for high cardinality to prevent huge confusion matrices
                    if y.nunique() > 20:
                        st.error(f"Target '{target_col}' has {y.nunique()} unique values. This looks like a regression task. Please select a target with fewer classes.")
                        return

                    # Encode target labels
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                if task_type == "Classification":
                    gnb = GaussianNB()
                    gnb.fit(X_train, y_train)
                    y_pred = gnb.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Accuracy: {acc:.2f}")

                    st.subheader("Classification Report")
                    # Use original class names for the report
                    target_names = [str(cls) for cls in le.classes_]
                    report = classification_report(y_test, y_pred, target_names=target_names, labels=range(len(target_names)), output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred, labels=range(len(target_names)))
                    # Display confusion matrix as a heatmap
                    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
                    st.dataframe(cm_df.style.background_gradient(cmap="Blues"))
                
                elif task_type == "Regression":
                    lr = LinearRegression()
                    lr.fit(X_train, y_train)
                    y_pred = lr.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.success(f"Mean Squared Error: {mse:.2f}")
                    st.success(f"R2 Score: {r2:.2f}")

                    st.subheader("Actual vs Predicted")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
