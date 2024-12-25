from cerberus import Validator

def validate_data_schema(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]]) -> bool:
    """Validate the dataframe against the given schema."""
    validator = Validator(schema)
    data_dict = df.to_dict(orient='list')
    return validator.validate(data_dict)

