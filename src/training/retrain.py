# Local application imports
from src.common import get_data
from src.common import clean_data
from src.common import visit_features

import pandas as pd

lab, pharmacy, visits, dem, mfl, dhs, txcurr = get_data.get_data(prediction = False)

# Clean lab and pharmacy data
lab = clean_data.clean_lab(lab, start_date = "2020-01-01")
pharmacy = clean_data.clean_pharmacy(pharmacy, start_date = "2020-01-01", end_date = "2025-01-15")

# Clean visits data, but first merge with dem to get demographics
# Merge visits with demographics (dem) on the "key" column
visits = visits.merge(
    dem[['key', 'Sex', 'MaritalStatus', 'EducationLevel', 'Occupation', 
         'ARTOutcomeDescription', 'StartARTDate', 'DOB']],
    on='key',
    how='inner'  
)
visits = clean_data.clean_visits(visits, start_date = "2020-01-01", end_date = "2025-01-15")

visits = visit_features.prep_visit_features(visits)


print(pd.crosstab(visits['pregnant'], visits['sex'], dropna=False))
