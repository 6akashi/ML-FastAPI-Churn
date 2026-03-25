from services.ChurnDatasetModule import ChurnDatasetModule

def test_prepare_data_logic(sample_data):
    loader = ChurnDatasetModule()
    loader.data = sample_data
    
    x, y, num, cat = loader.prepare_data()
    
    assert x.shape[0] == 2
    assert "churn" not in x.columns
    assert len(num) == 6
    assert len(cat) == 3