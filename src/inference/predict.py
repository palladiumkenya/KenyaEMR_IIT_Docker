# Local application imports
from src.common.get_data import get_data
from src.common import clean_data
from src.common import visit_features
from src.common import dem_features
from src.common import create_target
from src.common import target_features

ppk = "7E14A8034F39478149EE6A4CA37A247C631D17907C746BE0336D3D7CEC68F66F"
sc = "13074"

def inference_pipeline(ppk, sc):
    """
    Run the inference pipeline for a given patientPK and sitecode.
    """
    # Get data
    lab, pharmacy, visits, dem, mfl, dhs, txcurr = get_data(prediction = True,
                                                            patientPK= ppk,
                                                            sitecode= sc)

    # Clean data
    lab = clean_data.clean_lab(lab, start_date = "2020-01-01")
    pharmacy = clean_data.clean_pharmacy(pharmacy, start_date = "2020-01-01", end_date = "2025-01-15")
    visits = clean_data.clean_visits(visits, dem, start_date = "2020-01-01", end_date = "2025-01-15")
    
    # Prepare features
    visits = visit_features.prep_visit_features(visits)
    visits = dem_features.prep_demographics(visits)
    
    # Create target
    targets = create_target.create_target(visits, pharmacy, dem)
    
    # Prepare target features
    targets = target_features.prep_target_visit_features(targets, visits)
    targets = target_features.prep_target_lab_features(targets, lab)
    targets = target_features.prep_target_pharmacy_features(targets, pharmacy)
    # Return the prepared targets
    return targets

# Run the inference pipeline
if __name__ == "__main__":
    targets = inference_pipeline(ppk, sc)
    print(targets.head())