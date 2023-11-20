"""
Input - tashdata.xlsx
Output- Performs Feature Engineering on the dataset and saves int_data.csv 
"""
import pandas as pd

def check(string, pos):
    if string[pos]=='C':
        return 1
    else:
        return 0

data_ = pd.read_excel('Data/tashdata.xlsx')
prep_data = data_[['y_analysisGender2Category', 'y_analysisGender3Category']].rename(columns=
                {'y_analysisGender2Category':'Gender2', 'y_analysisGender3Category':'Gender3'})
CV_theme = data_['p_analysisThemeCV']
prep_data['CountC'] = CV_theme.map(lambda x: str(x).count('C'))
prep_data['CountV'] = CV_theme.map(lambda x: str(x).count('V'))
prep_data['Start']  = CV_theme.map(lambda x: check(str(x), 0))
prep_data['End']  = CV_theme.map(lambda x: check(str(x), -1))
prep_data['start_letter'] = data_['p_analysisThemeInitialSegment']
prep_data['end_letter'] = data_['p_analysisThemeFinalSegment']
prep_data['plu_pattern'] = data_['m_analysisPluralPattern']
prep_data['aug_vowel'] = data_['m_analysisRAugVowel']
prep_data['source'] = data_['m_lexiconLoanwordSource']
prep_data['derived_cat'] = data_['m_wordDerivedCategory']
prep_data['semantic_cat'] = data_['m_wordNumSemanticCategory']
prep_data['human'] = data_['s_lexiconHumanYN']
prep_data['animate'] = data_['s_lexiconAnimateYN']
prep_data['semantic'] = data_['s_lexiconSemanticField']
prep_data['sexgen'] = data_['s_lexiconSexGender']
prep_data.to_csv('Data/int_data.csv')