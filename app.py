import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

df_1 = pd.read_csv("first_telc.csv")

q = ""


@app.route("/")
def loadPage():
    return render_template("home.html", query="")


@app.route("/", methods=["POST"])
def predict():
    """
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    """

    inputQuery1 = request.form["query1"]
    inputQuery2 = request.form["query2"]
    inputQuery3 = request.form["query3"]
    inputQuery4 = request.form["query4"]
    inputQuery5 = request.form["query5"]
    inputQuery6 = request.form["query6"]
    inputQuery7 = request.form["query7"]
    inputQuery8 = request.form["query8"]
    inputQuery9 = request.form["query9"]
    inputQuery10 = request.form["query10"]
    inputQuery11 = request.form["query11"]
    inputQuery12 = request.form["query12"]
    inputQuery13 = request.form["query13"]
    inputQuery14 = request.form["query14"]
    inputQuery15 = request.form["query15"]
    inputQuery16 = request.form["query16"]
    inputQuery17 = request.form["query17"]
    inputQuery18 = request.form["query18"]
    inputQuery19 = request.form["query19"]

    model = pickle.load(open("model_new.sav", "rb"))

    data = [
        [
            inputQuery1,
            inputQuery2,
            inputQuery3,
            inputQuery4,
            inputQuery5,
            inputQuery6,
            inputQuery7,
            inputQuery8,
            inputQuery9,
            inputQuery10,
            inputQuery11,
            inputQuery12,
            inputQuery13,
            inputQuery14,
            inputQuery15,
            inputQuery16,
            inputQuery17,
            inputQuery18,
            inputQuery19,
        ]
    ]

    new_df = pd.DataFrame(
        data,
        columns=[
            "SeniorCitizen",
            "MonthlyCharges",
            "TotalCharges",
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "tenure",
        ],
    )

    # Convert appropriate columns to numeric values
    new_df["TotalCharges"] = pd.to_numeric(new_df["TotalCharges"], errors="coerce")
    new_df["tenure"] = pd.to_numeric(new_df["tenure"], errors="coerce")

    # Handle missing values in the new input data
    new_df["TotalCharges"].fillna(0, inplace=True)
    new_df["tenure"].fillna(0, inplace=True)

    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Ensure that tenure is numeric in the combined DataFrame
    df_2["tenure"] = pd.to_numeric(df_2["tenure"], errors="coerce")
    df_2["tenure"].fillna(0, inplace=True)

    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

    df_2["tenure_group"] = pd.cut(
        df_2["tenure"].astype(int), range(1, 80, 12), right=False, labels=labels
    )
    # drop column customerID and tenure
    df_2.drop(columns=["tenure"], axis=1, inplace=True)

    new_df_dummies = pd.get_dummies(
        df_2[
            [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "tenure_group",
            ]
        ]
    )

    # Remove duplicate columns
    new_df_dummies = new_df_dummies.loc[:, ~new_df_dummies.columns.duplicated()]

    # Ensure that the new data has the same dummy variables as the training data
    new_df_dummies = new_df_dummies.reindex(
        columns=model.feature_names_in_, fill_value=0
    )

    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]

    if single == 1:
        o1 = "This customer is likely to churn!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)

    return render_template(
        "home.html",
        output1=o1,
        output2=o2,
        query1=request.form["query1"],
        query2=request.form["query2"],
        query3=request.form["query3"],
        query4=request.form["query4"],
        query5=request.form["query5"],
        query6=request.form["query6"],
        query7=request.form["query7"],
        query8=request.form["query8"],
        query9=request.form["query9"],
        query10=request.form["query10"],
        query11=request.form["query11"],
        query12=request.form["query12"],
        query13=request.form["query13"],
        query14=request.form["query14"],
        query15=request.form["query15"],
        query16=request.form["query16"],
        query17=request.form["query17"],
        query18=request.form["query18"],
        query19=request.form["query19"],
    )


if __name__ == "__main__":
    app.run(debug=True)
