import pandas as pd
import numpy as np
import pickle
def create_categorical_labels (full_path):
    df=pd.read_excel(full_path)
    #drop rows where there is no outcome
    df.drop(df[(df['Summary'].isnull()) ].index, inplace=True)
    key_finding_columns=df.columns[11:]
    #create new columns with all outcomes concatenated
    def join_not_nan(list_elements):
        return "_".join([x.lower() for x in list_elements if type(x) == str])
    df["all_outcomes"] = df[key_finding_columns].agg(join_not_nan, axis=1)
    possible_outcomes=['Medias', 'Pleura', 'Diaphragm', 'Device', 'Other', 'GI', 'PNX', 'Cardiac', 'Lung', 'Bone', 'Vascular']
    for outcome in possible_outcomes:
        def get_outcome(element):
            return 1 if outcome.lower() in element.split("_") else 0
        df[outcome] = df["all_outcomes"].apply(get_outcome)
    def getNormal(element):
        return 1 if str(element).lower() == "normal" else 0
    df['Normal'] = df["Summary"].apply(getNormal)
    return df

if __name__=="__main__":
    df=create_categorical_labels("C:\\Users\\ngozzi\\Documents\\Progetti_HUM\\X-ray\\X-ray\\X_ray_database_py.xls")
    print(df.head())
    df.to_pickle("Xray_database_py_correctlabels.pkl")