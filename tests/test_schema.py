"""Schema test: pandera valida o parquet limpo."""

import pandera.pandas as pa
import pandas as pd
from pandera.pandas import Column, DataFrameSchema, Check

TELCO_CLEAN_SCHEMA = DataFrameSchema(
    {
        "customerID": Column(str, unique=True),
        "gender": Column(str, Check.isin(["Female", "Male"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.in_range(0, 120)),
        "PhoneService": Column(str, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(str, Check.isin(["Yes", "No"])),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(str, Check.isin(["Yes", "No"])),
        "OnlineBackup": Column(str, Check.isin(["Yes", "No"])),
        "DeviceProtection": Column(str, Check.isin(["Yes", "No"])),
        "TechSupport": Column(str, Check.isin(["Yes", "No"])),
        "StreamingTV": Column(str, Check.isin(["Yes", "No"])),
        "StreamingMovies": Column(str, Check.isin(["Yes", "No"])),
        "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": Column(float, Check.in_range(0, 500), nullable=False),
        "TotalCharges": Column(float, Check.in_range(0, 40000), nullable=False),
        "Churn": Column(str, Check.isin(["Yes", "No"])),
    },
    strict=True,  # rejeita coluna nao declarada
    coerce=False,
)


def test_clean_parquet_matches_schema(clean_dataset_path) -> None:
    """O parquet limpo respeita o contrato pos-EDA."""
    df = pd.read_parquet(clean_dataset_path)

    # Validacao principal: nao levanta SchemaError
    validated = TELCO_CLEAN_SCHEMA.validate(df, lazy=True)
    assert len(validated) == len(df)

    # Sanity extras
    assert df["TotalCharges"].isna().sum() == 0  
    assert df["MultipleLines"].isin(["Yes", "No"]).all()