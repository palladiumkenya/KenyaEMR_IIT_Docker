# Local application imports
from src.common import get_data
from src.common import clean_data
from src.common import visit_features
from src.common import dem_features
from src.common import create_target
from src.common import target_features

lab, pharmacy, visits, dem, mfl, dhs, txcurr = get_data.get_data(prediction = False)

# Run cleaning and feature preparation functions
lab = clean_data.clean_lab(lab, start_date = "2020-01-01")
pharmacy = clean_data.clean_pharmacy(pharmacy, start_date = "2020-01-01", end_date = "2025-01-15")
visits = clean_data.clean_visits(visits, dem, start_date = "2020-01-01", end_date = "2025-01-15")
visits = visit_features.prep_visit_features(visits)
visits = dem_features.prep_demographics(visits)
targets = create_target.create_target(visits, pharmacy, dem)
targets = target_features.prep_target_visit_features(targets, visits)
targets = target_features.prep_target_pharmacy_features(targets, pharmacy)
targets = target_features.prep_target_lab_features(targets, lab)
