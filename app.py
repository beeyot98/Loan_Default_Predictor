import pandas as pd
import numpy as np
import streamlit as st
import pickle

app_mode = st.sidebar.selectbox('Select Page',['Home','Predict_Default'])

if app_mode == "Home":
    st.title("Loan Default Prediction")
    st.markdown('Dataset :')
    df=pd.read_csv('archive/test.csv')
    st.write(df.head())
elif app_mode== "Predict_Default":
    st.subheader("Fill in loan details")
    st.sidebar.header("Other details:")

    cns_score = {'Very High Risk': 5, 'High Risk': 4, 'Medium Risk': 3, 'Low Risk': 2, 'Very Low Risk': 1, 'No Score': 0}
    DISBURSED_AMOUNT=st.number_input('DISBURSED_AMOUNT')
    ASSET_COST = st.number_input('ASSET_COST')
    NO_OF_INQUIRIES= st.number_input('NO_OF_INQUIRIES')
    CREDIT_HISTORY_LENGTH = st.number_input('CREDIT_HISTORY_LENGTH')
    APPLICANT_AGE= st.number_input('APPLICANT_AGE')
    PERFORM_CNS_SCORE_KEY = st.sidebar.radio("Select CNS_rating ",tuple(cns_score.keys()))
    PERFORM_CNS_SCORE_DESCRIPTION = cns_score.get(PERFORM_CNS_SCORE_KEY)
    LTV=st.number_input('LTV')
    PRI_NO_OF_ACCTS=st.number_input('PRI_NO_OF_ACCTS')
    PRI_OVERDUE_ACCTS=st.number_input('PRI_OVERDUE_ACCTS')
    PRI_CURRENT_BALANCE=st.number_input('PRI_CURRENT_BALANCE')
    PRI_SANCTIONED_AMOUNT=st.number_input('PRI_SANCTIONED_AMOUNT')
    PRIMARY_INSTAL_AMT=st.number_input('PRIMARY_INSTAL_AMT')
    NEW_ACCTS_IN_LAST_SIX_MONTHS=st.number_input('NEW_ACCTS_IN_LAST_SIX_MONTHS')
    DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS=st.number_input('DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS')
    employ = {'Salaried': 1, 'Self_employed': 2, 'Missing': 3}
    EMPLOYMENT_TYPE= st.sidebar.radio("Select Employement Status",tuple(employ.keys()))
    CUSTOMER_FLAG = st.multiselect('ID presented', ('AADHAR', 'PAN', 'VOTERID', 'DRIVING',
       'PASSPORT'))
    Salaried,Self_employed,Missing=0,0,0
    if EMPLOYMENT_TYPE == 'Salaried':
        Salaried = 1
    elif EMPLOYMENT_TYPE == 'Self Employed':
        Self_employed = 1
    else :Missing = 0

    AADHAR_FLAG,PAN_FLAG,VOTERID_FLAG,DRIVING_FLAG,PASSPORT_FLAG=0,0,0,0,0
    if CUSTOMER_FLAG == 'AADHAR':
        AADHAR_FLAG=1
    else:
        AADHAR_FLAG=0
    if CUSTOMER_FLAG == 'PAN':
        PAN_FLAG = 1
    else:
        PAN_FLAG=0
    if CUSTOMER_FLAG == 'VOTERDID':
        VOTERID_FLAG = 1
    else:
        VOTERID_FLAG=0
    if CUSTOMER_FLAG =='DRIVING':
        DRIVING_FLAG= 1
    else:
        DRIVING_FLAG=0
    if CUSTOMER_FLAG== 'PASSPORT':
        PASSPORT_FLAG = 1
    else:
        PASSPORT_FLAG = 0

    subdata = {'DISBURSED_AMOUNT':DISBURSED_AMOUNT,'ASSET_COST':ASSET_COST,'NO_OF_INQUIRIES':NO_OF_INQUIRIES,'CREDIT_HISTORY_LENGTH':CREDIT_HISTORY_LENGTH,'APPLICANT_AGE':APPLICANT_AGE, 'PERFORM_CNS_SCORE_DESCRIPTION':PERFORM_CNS_SCORE_DESCRIPTION, 'LTV':LTV,'PRI_NO_OF_ACCTS':PRI_NO_OF_ACCTS,
    'PRI_CURRENT_BALANCE':PRI_CURRENT_BALANCE,'PRI_OVERDUE_ACCTS':PRI_OVERDUE_ACCTS,'PRI_SANCTIONED_AMOUNT':PRI_SANCTIONED_AMOUNT,'PRIMARY_INSTAL_AMT':PRIMARY_INSTAL_AMT,
    'NEW_ACCTS_IN_LAST_SIX_MONTHS':NEW_ACCTS_IN_LAST_SIX_MONTHS,'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS':DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS,'AADHAR_FLAG':AADHAR_FLAG,'PAN_FLAG':PAN_FLAG,'VOTERID_FLAG':VOTERID_FLAG,'DRIVING_FLAG':DRIVING_FLAG,'PASSPORT_FLAG':PASSPORT_FLAG,'EMPLOYMENT_TYPE':[Salaried,Self_employed,Missing]}
    features = [DISBURSED_AMOUNT,ASSET_COST,NO_OF_INQUIRIES,CREDIT_HISTORY_LENGTH,APPLICANT_AGE, PERFORM_CNS_SCORE_DESCRIPTION,LTV,PRI_SANCTIONED_AMOUNT,PRI_CURRENT_BALANCE,PRI_NO_OF_ACCTS,PRI_OVERDUE_ACCTS,PRIMARY_INSTAL_AMT,NEW_ACCTS_IN_LAST_SIX_MONTHS,DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS,AADHAR_FLAG,PAN_FLAG,VOTERID_FLAG,DRIVING_FLAG,PASSPORT_FLAG, subdata['EMPLOYMENT_TYPE'][0],subdata['EMPLOYMENT_TYPE'][1], subdata['EMPLOYMENT_TYPE'][2]]
    results = np.array(features).reshape(1, -1)

if st.button("Predict"):

    picklefile = open("finalized_model.sav", "rb")
    model = pickle.load(picklefile)

    prediction = model.predict(results)
    if prediction[0] == 0:
        st.success('Customer will not default')
    elif prediction[0] == 1:
        st.error( 'Customer will default')
        
streamlit run app.py
