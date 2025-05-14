# Local application imports
from src.common.get_data import get_data
from src.common import clean_data
from src.common import visit_features
from src.common import dem_features
from src.common import create_target
from src.common import target_features

ppk = "7E14A8034F39478149EE6A4CA37A247C631D17907C746BE0336D3D7CEC68F66F"
sc = "13074"

lab, pharmacy, visits, dem, mfl, dhs, txcurr = get_data(prediction = True,
                                                        patientPK= ppk,
                                                        sitecode= sc)

lab = clean_data.clean_lab(lab, start_date = "2020-01-01")
pharmacy = clean_data.clean_pharmacy(pharmacy, start_date = "2020-01-01", end_date = "2025-01-15")
visits = clean_data.clean_visits(visits, dem_df, start_date = "2020-01-01", end_date = "2025-01-15")
visits = visit_features.prep_visit_features(visits)
visits = dem_features.prep_demographics(visits)
targets = create_target.create_target(visits, pharmacy, dem)
targets = target_features.prep_target_visit_features(targets, visits)