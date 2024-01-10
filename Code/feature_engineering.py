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
prep_data['p_lengthC'] = CV_theme.map(lambda x: str(x).count('C'))
prep_data['p_lengthV'] = CV_theme.map(lambda x: str(x).count('V'))
prep_data['p_startCorV']  = CV_theme.map(lambda x: check(str(x), 0))
prep_data['p_endCorV']  = CV_theme.map(lambda x: check(str(x), -1))
prep_data['p_startPhoneme'] = data_['p_analysisThemeInitialSegment']
prep_data['p_endPhoneme'] = data_['p_analysisThemeFinalSegment']
prep_data['m_pluralPattern'] = data_['m_analysisPluralPattern']
prep_data['m_rAugVowel'] = data_['m_analysisRAugVowel']
prep_data['m_loanwordSource'] = data_['m_lexiconLoanwordSource']
prep_data['m_derivaionalCat'] = data_['m_wordDerivedCategory']
prep_data['m_secondaryMorph'] = data_['m_wordNumSemanticCategory']
prep_data['s_humanYN'] = data_['s_lexiconHumanYN']
prep_data['s_animateYN'] = data_['s_lexiconAnimateYN']
prep_data['s_semanticField'] = data_['s_lexiconSemanticField']
prep_data['s_SexGender'] = data_['s_lexiconSexGender']
prep_data.to_csv('Data/int_data.csv')