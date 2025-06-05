import pandas as pd
from src.common.create_target import create_target

def test_create_target_minimal():
    # make sure visitdaet and nad_imputed are in datetime format
    visits = pd.DataFrame({
        "key": ["A", "A"],
        "visitdate": ["2022-01-01", "2022-07-01"],
        "nad_imputed": ["2022-07-01", "2022-10-01"],
        "nad_imputation_flag": [0, 0],
        "sitecode": ["001", "001"]
    })
    visits["visitdate"] = pd.to_datetime(visits["visitdate"])
    visits["nad_imputed"] = pd.to_datetime(visits["nad_imputed"])
    # make sure nad_imputed is in datetime format
    pharmacy = pd.DataFrame({
        "key": ["A", "A"],
        "dispensedate": ["2022-01-01", "2022-07-01"],
        "nad_imputed": ["2022-05-01", "2022-11-01"],
        "nad_imputation_flag": [0, 0],
        "sitecode": ["001", "001"]
    })
    pharmacy["dispensedate"] = pd.to_datetime(pharmacy["dispensedate"])
    pharmacy["nad_imputed"] = pd.to_datetime(pharmacy["nad_imputed"])
    dem = pd.DataFrame({
        "key": ["A"],
        "artoutcomedescription": ["active"]
    })
    result = create_target(visits, pharmacy, dem)
    assert "key" in result.columns
    assert "visitdate" in result.columns
    assert "nad" in result.columns
    assert "iit" in result.columns
    assert result.shape[0] == 1


def test_create_target_deduplication():
    visits = pd.DataFrame({
        "key": ["A", "A"],
        "visitdate": ["2022-01-01", "2022-07-01"],
        "nad_imputed": ["2022-07-01", "2022-10-01"],
        "nad_imputation_flag": [1, 0],
        "sitecode": ["001", "001"]
    })
    visits["visitdate"] = pd.to_datetime(visits["visitdate"])
    visits["nad_imputed"] = pd.to_datetime(visits["nad_imputed"])
    pharmacy = pd.DataFrame({
        "key": ["A"],
        "dispensedate": ["2022-01-01"],
        "nad_imputed": ["2022-01-03"],
        "nad_imputation_flag": [0],
        "sitecode": ["001"]
    })
    pharmacy["dispensedate"] = pd.to_datetime(pharmacy["dispensedate"])
    pharmacy["nad_imputed"] = pd.to_datetime(pharmacy["nad_imputed"])
    dem = pd.DataFrame({
        "key": ["A"],
        "artoutcomedescription": ["active"]
    })
    result = create_target(visits, pharmacy, dem)
    print(result)
    # Should deduplicate and keep the row with nad_imputation_flag 0 and latest nad
    assert result.shape[0] == 1
    assert result.iloc[0]["nad"] == pd.Timestamp("2022-01-03")

def test_create_target_iit_logic():
    visits = pd.DataFrame({
        "key": ["A", "A"],
        "visitdate": ["2022-01-01", "2022-04-10"],
        "nad_imputed": ["2022-03-01", "2022-07-10"],
        "nad_imputation_flag": [0, 0],
        "sitecode": ["001", "001"]
    })
    visits["visitdate"] = pd.to_datetime(visits["visitdate"])
    visits["nad_imputed"] = pd.to_datetime(visits["nad_imputed"])
    pharmacy = pd.DataFrame({
        "key": ["A"],
        "dispensedate": ["2022-01-01"],
        "nad_imputed": ["2022-03-01"],
        "nad_imputation_flag": [0],
        "sitecode": ["001"]
    })
    pharmacy["dispensedate"] = pd.to_datetime(pharmacy["dispensedate"])
    pharmacy["nad_imputed"] = pd.to_datetime(pharmacy["nad_imputed"])
    dem = pd.DataFrame({
        "key": ["A"],
        "artoutcomedescription": ["active"]
    })
    result = create_target(visits, pharmacy, dem)
    # The first visit resulted in iit (gap > 30 days)
    assert result[result["visitdate"] == "2022-01-01"]["iit"].iloc[0] == 1


def test_create_target_artoutcomedescription_filter():
    visits = pd.DataFrame({
        "key": ["A", "A", "B"],
        "visitdate": ["2022-01-01", "2022-04-10", "2022-12-01"],
        "nad_imputed": ["2022-03-01", "2022-07-10", "2023-01-01"],
        "nad_imputation_flag": [0, 0, 0],
        "sitecode": ["001", "001", "001"]
    })
    visits["visitdate"] = pd.to_datetime(visits["visitdate"])
    visits["nad_imputed"] = pd.to_datetime(visits["nad_imputed"])
    pharmacy = pd.DataFrame({
        "key": ["A"],
        "dispensedate": ["2022-01-01"],
        "nad_imputed": ["2022-03-01"],
        "nad_imputation_flag": [0],
        "sitecode": ["001"]
    })
    pharmacy["dispensedate"] = pd.to_datetime(pharmacy["dispensedate"])
    pharmacy["nad_imputed"] = pd.to_datetime(pharmacy["nad_imputed"])
    dem = pd.DataFrame({
        "key": ["A"],
        "artoutcomedescription": ["died"]
    })
    result = create_target(visits, pharmacy, dem)
    print(result)
    # Should filter out because artoutcomedescription is died
    # if active, it should return 2 rows
    assert result.shape[0]==1

def test_create_target_unresolved_outcome():
    visits = pd.DataFrame({
        "key": ["A"],
        "visitdate": ["2022-01-01"],
        "nad_imputed": ["2022-01-01"],
        "nad_imputation_flag": [0],
        "sitecode": ["001"]
    })
    visits["visitdate"] = pd.to_datetime(visits["visitdate"])
    visits["nad_imputed"] = pd.to_datetime(visits["nad_imputed"])
    pharmacy = pd.DataFrame({
        "key": ["A"],
        "dispensedate": ["2022-01-01"],
        "nad_imputed": ["2022-01-01"],
        "nad_imputation_flag": [0],
        "sitecode": ["001"]
    })
    pharmacy["dispensedate"] = pd.to_datetime(pharmacy["dispensedate"])
    pharmacy["nad_imputed"] = pd.to_datetime(pharmacy["nad_imputed"])
    dem = pd.DataFrame({
        "key": ["A"],
        "artoutcomedescription": ["active"]
    })
    # fudge max_visitdate so that nad+30 > max_visitdate
    # (simulate by making only one visit)
    result = create_target(visits, pharmacy, dem)
    # Should be empty because outcome is unresolved
    assert result.empty