# Local application imports
from src.common import get_data
from src.common import clean_data

lab, pharmacy, visits, dem, mfl, dhs, txcurr = get_data.get_data(prediction = False)

# Clean lab data
lab = clean_data.clean_lab(lab, start_date = "2020-01-01")