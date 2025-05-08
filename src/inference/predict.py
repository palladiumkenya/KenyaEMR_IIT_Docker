# Local application imports
from src.common.get_data import get_data

ppk = "7E14A8034F39478149EE6A4CA37A247C631D17907C746BE0336D3D7CEC68F66F"
sc = "13074"

lab, pharmacy, visits, dem, mfl, dhs, txcurr = get_data(prediction = True,
                                                        patientPK= ppk,
                                                        sitecode= sc)