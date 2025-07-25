# Local application imports
from src.inference import get_inference_data
from src.common import clean_data
from src.common import visit_features
from src.common import dem_features
from src.common import create_target
from src.common import target_features
from src.inference import locational_features_inf
from src.inference import generate_inference

def run_inference_pipeline(ppk = str, sc = str, start_date = str, end_date = str):

    # For retraining, prediction is False, so won't add that as argument to parent function
    # lab, pharmacy, visits, dem = get_inference_data.get_inference_data_sqlite(patientPK= ppk, sitecode= sc)
    lab, pharmacy, visits, dem = get_inference_data.get_inference_data_mysql(patientPK= ppk, sitecode= sc)

    # Run cleaning and feature preparation functions
    lab = clean_data.clean_lab(lab, start_date = start_date)
    pharmacy = clean_data.clean_pharmacy(pharmacy, start_date = start_date, end_date = end_date)
    visits = clean_data.clean_visits(visits, dem, start_date = start_date, end_date = end_date)
    visits = visit_features.prep_visit_features(visits)
    visits = dem_features.prep_demographics(visits)
    targets = create_target.create_target(visits, pharmacy, dem)
    targets = target_features.prep_target_visit_features(targets, visits)
    targets = target_features.prep_target_pharmacy_features(targets, pharmacy)
    targets = target_features.prep_target_lab_features(targets, lab)
    targets = locational_features_inf.get_locational_features(targets)
    pred = generate_inference.gen_inference(targets, sc)
    print(pred)
    return pred

if __name__ == "__main__":
    run_inference_pipeline(ppk = "7E14A8034F39478149EE6A4CA37A247C631D17907C746BE0336D3D7CEC68F66F",
                           sc = "13074",
                            start_date = "2021-01-01",
                            end_date = "2025-01-15")