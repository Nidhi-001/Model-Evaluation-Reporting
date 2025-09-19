import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, precision_recall_curve, auc,root_mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import numpy as np
# from pdf import save_evaluation_report



# 0. Timestamp for this run

timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


# 1. Metrics Functions
def evaluate_classification(y_true, y_pred, y_proba=None):
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        # "RMSE": root_mean_squared_error(y_true,y_proba),
        # "MAE": mean_absolute_error(y_true,y_proba)
    }
    if y_proba is not None:
        results["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    return results


# 2. Plotting Functions
def plot_roc_curve(y_true,y_proba,model_name,timestamp):
    fpr,tpr,_=roc_curve(y_true,y_proba)
    
    plt.plot(fpr,tpr,label=f"{model_name}(AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"figs/roc_{model_name}_{timestamp}.png")
    plt.clf()  # clf :  clear figure (reuse same figure)
    
    # plt.show()      # close: close figure (delete the figure)
    # plt.close()                 


def plot_pr_curve(y_true, y_proba, model_name, timestamp):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(f"figs/pr_{model_name}_{timestamp}.png")
    plt.clf()

def plot_confusion(y_true, y_pred, model_name, timestamp):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(f"figs/cm_{model_name}_{timestamp}.png")
    plt.clf()   #to avoid overlapp of plots as matplotlib saves the plot in memory


# 3. Model Runner
def run_classification_model(model, X_train, X_test, y_train, y_test, model_name, timestamp):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    
    metrics = evaluate_classification(y_test, y_pred, y_proba)

    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, model_name, timestamp)
        plot_pr_curve(y_test, y_proba, model_name, timestamp)
        # save_evaluation_report(y_test, y_pred, y_proba, "breast_cancer_eval.pdf")

    plot_confusion(y_test, y_pred, model_name, timestamp)
    
    

    return metrics


# 4. Save Model Card

def save_model_card(model_name, dataset_name, task, metrics, filename):
    with open(filename, "w", encoding="utf-8") as f:  # UTF-8 for emojis
        f.write(f"# Model Card: {model_name}\n\n")
        f.write("## Model Details\n")
        f.write(f"- **Name**: {model_name}\n")
        f.write(f"- **Task**: {task}\n")
        f.write("- **Library**: scikit-learn\n\n")
        f.write("## Dataset\n")
        f.write(f"- **Name**: {dataset_name}\n")
        f.write(f"- **Type**: {task}\n\n")
        f.write("## Evaluation Metrics\n")
        for k, v in metrics.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\n")
        f.write("## Intended Use\n- Educational demo\n- Benchmark comparisons\n\n")
        f.write("## Limitations\n- Sensitive to hyperparameters\n- Performance depends on dataset size/quality\n\n")
        f.write("## Ethical Considerations\n- Use responsibly, especially for sensitive data.\n")



# 5. Save HTML Report
def save_html_report(results, filename,timestamp):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report.html")
    html_out = template.render(results=results,timestamp=timestamp)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"✅ HTML report saved: {filename}")



    
# 6. Main Script

if __name__ == "__main__":
    # Make folders
    os.makedirs("figs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("model_cards", exist_ok=True)

    # Load dataset
    data = load_breast_cancer()
   
    # x = data.data
    # y = data.target
    # feature_names = data.feature_names
    # target_names = data.target_names
    # d1 = pd.DataFrame(x)
    # d1['target'] = y
    # print(d1.head)

    
    p=pd.DataFrame(data.data,columns=data.feature_names)
    q=pd.DataFrame(data.target)
    print(p.head())
    print(p.info())
    print(p.shape)
    # p.to_csv("breast_cancer_dataset.csv",index=False)
    print(q.head())
    print(data['target_names'])
    print(q.nunique())
    print(q.value_counts())    #60-40 ratio target column ..imbalance dataset so f1 score is calculated
    print(q.info())
    # q.to_csv("target.csv",index=False)
    # print(data)
    
    
    


    
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.1, random_state=42
    )   #stratify=for imbalanced dataset(test data has more malignant cases in train, less in test)
    print(X_test.shape)
    
    

    # Define models
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=50))
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics = run_classification_model(model, X_train, X_test, y_train, y_test, name, timestamp)
        results[name] = metrics

        # Save model card
        save_model_card(
            model_name=name,
            dataset_name="Breast Cancer",
            task="Classification",
            metrics=metrics,
            filename=f"model_cards/{name}_{timestamp}.md"
        )

       

    # Save metrics CSV
    df = pd.DataFrame(results).T
    # df.to_csv(f"reports/metrics_{timestamp}.csv")
    print("\nFinal Results:\n", df)


    # Save HTML report
    save_html_report(results, filename=f"reports/final_report_{timestamp}.html",timestamp=timestamp)

   

    
    
    