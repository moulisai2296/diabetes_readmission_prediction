"""Request / response schemas for the readmission-risk API.

The request is the *post-ETL* encounter: the standardized field values a hospital's
integration layer would supply from the EHR at discharge (the same level as
`cleaned.parquet`). The server then does all feature engineering — the caller never
computes engineered features, which is what keeps training and serving in lockstep.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgeBracket = Literal[
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
MedStatus = Literal["No", "Steady", "Up", "Down"]

EXAMPLE_ENCOUNTER = {
    "race": "Caucasian", "gender": "Female", "age": "[70-80)",
    "time_in_hospital": 9, "payer_code": "Missing", "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 25, "num_procedures": 3, "num_medications": 16,
    "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 2,
    "diag_1": "428", "diag_2": "427", "diag_3": "250.01", "number_diagnoses": 7,
    "max_glu_serum": "NotMeasured", "A1Cresult": "NotMeasured",
    "glyburide": "Steady", "insulin": "Down",
    "change": "Ch", "diabetesMed": "Yes",
    "admission_type": "Elective",
    "discharge_disposition": "Discharged/transferred to another type of inpatient care institution",
    "admission_source": "Transfer from a hospital",
}


class PatientEncounter(BaseModel):
    """One hospital stay at the moment of discharge. Medication fields default to
    'No' (not prescribed / unchanged), the common case, so callers send only what
    applies. All values must be the cleaned/standardized forms (e.g. 'Missing' for
    unknown payer, 'NotMeasured' for an absent lab)."""

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={"example": EXAMPLE_ENCOUNTER})

    race: str = "Missing"
    gender: Literal["Male", "Female"]
    age: AgeBracket

    time_in_hospital: int = Field(ge=1, le=14)
    payer_code: str = "Missing"
    medical_specialty: str = "Missing"
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    number_diagnoses: int = Field(ge=1)

    diag_1: str | None = None
    diag_2: str | None = None
    diag_3: str | None = None

    max_glu_serum: Literal[">200", ">300", "Norm", "NotMeasured"] = "NotMeasured"
    A1Cresult: Literal[">7", ">8", "Norm", "NotMeasured"] = "NotMeasured"

    metformin: MedStatus = "No"
    repaglinide: MedStatus = "No"
    nateglinide: MedStatus = "No"
    chlorpropamide: MedStatus = "No"
    glimepiride: MedStatus = "No"
    acetohexamide: MedStatus = "No"
    glipizide: MedStatus = "No"
    glyburide: MedStatus = "No"
    tolbutamide: MedStatus = "No"
    pioglitazone: MedStatus = "No"
    rosiglitazone: MedStatus = "No"
    acarbose: MedStatus = "No"
    miglitol: MedStatus = "No"
    troglitazone: MedStatus = "No"
    tolazamide: MedStatus = "No"
    insulin: MedStatus = "No"
    glyburide_metformin: MedStatus = Field("No", alias="glyburide-metformin")
    glipizide_metformin: MedStatus = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: MedStatus = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedStatus = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedStatus = Field("No", alias="metformin-pioglitazone")

    change: Literal["Ch", "No"] = "No"
    diabetesMed: Literal["Yes", "No"] = "No"
    admission_type: str = "Unknown"
    discharge_disposition: str = "Unknown"
    admission_source: str = "Unknown"


class Factor(BaseModel):
    feature: str
    value: str
    contribution: float = Field(description="signed; positive pushes risk up")
    direction: Literal["increases", "decreases"]


class PredictionResponse(BaseModel):
    readmission_risk: float = Field(description="calibrated probability of <30-day readmission")
    flagged: bool = Field(description="risk >= operating threshold -> recommend follow-up")
    threshold: float
    model_version: str
    top_factors: list[Factor]
