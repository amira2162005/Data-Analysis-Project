
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import sqlite3 

# _______________________________________________________________________________________
def load_data():
    data = pd.read_csv('/usr/local/airflow/include/Final_data.csv')

def check_data():
    
    data = pd.read_csv('/usr/local/airflow/include/Final_data.csv')
    print("Dimensions:", data.shape)
    print("Null values:\n", data.isnull().sum())
    print("\nInformation about DATA: ")
    print(data.info())
    print("\nHead of DATA: \n", data.head())


def clean_data():
    data = pd.read_csv('/usr/local/airflow/include/Final_data.csv')
   
    
    data.drop_duplicates(keep='first', inplace=True)

    data['Contract Valid Until'] = data['Contract Valid Until'].fillna(2025)
    data['Composure'] = data['Composure'].fillna(data['Composure'].mean())
    
    data.to_csv("/usr/local/airflow/include/cleaned_data.csv", index=False)
    


# _______________________________________________________________________________________

def Visualize_data():
    data = pd.read_csv('/usr/local/airflow/include/Final_data.csv')
    data_cleaned = pd.read_csv("/usr/local/airflow/include/cleaned_data.csv")

    fig = px.histogram(data_cleaned, x="Age", nbins=10, title="Histogram of Age column")
    fig.update_layout(xaxis_title="Age", yaxis_title="Frequency")
    fig.show()

    fig = px.scatter(data_cleaned, x="BallControl", y="Skill Moves", color="Age", title="Relationship between Ball Control and Skill Moves")
    fig.show()

    fig = px.box(data, y="Skill Moves", title="Box Plot for Skill Moves")
    fig.update_layout(xaxis_title="Features", yaxis_title="Values")
    fig.show()

    fig = px.histogram(data_cleaned, x="Best Overall Rating", nbins=25, title="Distribution of Best Overall Rating")
    fig.show()


    club_counts = data_cleaned['Preferred Foot'].value_counts()
    fig = px.pie(names=club_counts.index, values=club_counts.values, title="Distribution of Players by Preferred Foot")
    fig.show()


    fig = px.histogram(data_cleaned, x="Work Rate", title="Work Rate Count")
    fig.show()


    selected_data = data_cleaned[['Skill Moves', 'ShortPassing', 'BallControl', 'ShotPower']]
    correlation_matrix = selected_data.corr()
    fig = px.imshow(correlation_matrix, title="Correlation Heatmap")
    fig.show()


    data_selected = data_cleaned[["Age", "ID", "Best Overall Rating", "BallControl"]]
    fig = px.scatter_matrix(data_selected, dimensions=["Age", "ID", "Best Overall Rating", "BallControl"], title="Pair Grid")
    fig.show()

# _______________________________________________________________________________________



def apply_ml_model():
    data_cleaned = pd.read_csv("/usr/local/airflow/include/cleaned_data.csv")
    data_input = data_cleaned.drop(columns=['Skill Moves'])
    data_output = data_cleaned['Skill Moves']
    
    for column in data_input.select_dtypes(include=['object']).columns:
        data_input[column] = data_input[column].apply(str)
        label_encoder = LabelEncoder()
        data_input[column] = label_encoder.fit_transform(data_input[column])

    imputer = SimpleImputer(strategy='mean')
    data_input = imputer.fit_transform(data_input)

    X_train, X_test, y_train, y_test = train_test_split(data_input, data_output, test_size=0.30, random_state=0)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(solver='saga', max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy is: ", accuracy_score(y_test, y_pred))
    print("Precision is: ", precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Recall is: ", recall_score(y_test, y_pred, average='weighted', zero_division=0))
    print("F1 score is: ", f1_score(y_test, y_pred, average='weighted', zero_division=0))

    fig = px.histogram(x=y_pred, title="Distribution of Predicted Classes (Histogram)")
    fig.show()
# _______________________________________________________________________________________



def create_database():
    conn = sqlite3.connect('plyers.db')
    cursor = conn.cursor()
    conn.commit()
    conn.close()

def load_and_store_data():
    DATA = pd.read_csv("/usr/local/airflow/include/cleaned_data.csv")

    
    conn = sqlite3.connect('plyers.db')

    DATA.to_sql('players_data', conn, if_exists='replace', index=False)

    df_check = pd.read_sql('SELECT * FROM players_data', conn)
    print(df_check.head())

    conn.close()

# _______________________________________________________________________________________


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'start_date': datetime(2025, 5, 10),
}

dag = DAG(
    'Data_Pipeline',
    default_args=default_args,
    description='A data pipeline DAG for data processing, visualization, ML, and database loading',
    schedule="@daily",  
    catchup=False,
)





create_database_task = PythonOperator(
    task_id='create_database',
    python_callable=create_database,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

clean_data_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    dag=dag,
)

visualize_data_task = PythonOperator(
    task_id='visualize_data',
    python_callable=Visualize_data,
    dag=dag,
)



ml_pipeline_task = PythonOperator(
    task_id='ml_pipeline',
    python_callable=apply_ml_model,
    dag=dag,
)

store_data_task = PythonOperator(
    task_id='store_data_in_sqlite',
    python_callable=load_and_store_data,
    dag=dag,
)

load_data_task >> clean_data_task >> visualize_data_task  >> [ml_pipeline_task , create_database_task ] >> store_data_task  
