from decimal import Decimal


def decimal_field_string(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.00000001")))
