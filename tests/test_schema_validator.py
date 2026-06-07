from src.registry.tool_registry import ToolRegistry
from src.validation.schema_validator import SchemaValidator


def test_valid_tool_call_has_no_schema_issues() -> None:
    registry = ToolRegistry.from_file("data/tools.json")
    validator = SchemaValidator()

    issues = validator.validate_call(
        registry.get("add_data_package"),
        {"customer_id": "C001", "package_code": "DATA10"},
        "add_data_package",
    )

    assert issues == []


def test_missing_required_argument_is_reported() -> None:
    registry = ToolRegistry.from_file("data/tools.json")
    validator = SchemaValidator()

    issues = validator.validate_call(registry.get("add_data_package"), {"customer_id": "C001"}, "add_data_package")

    assert [issue.code for issue in issues] == ["missing_arg"]
    assert "package_code" in issues[0].message


def test_invalid_enum_is_reported() -> None:
    registry = ToolRegistry.from_file("data/tools.json")
    validator = SchemaValidator()

    issues = validator.validate_call(
        registry.get("enable_roaming"),
        {"customer_id": "C001", "country": "Atlantis"},
        "enable_roaming",
    )

    assert [issue.code for issue in issues] == ["invalid_enum"]


def test_unknown_tool_is_reported() -> None:
    validator = SchemaValidator()

    issues = validator.validate_call(None, {"customer_id": "C001"}, "legacy_enable_roaming")

    assert [issue.code for issue in issues] == ["unknown_tool"]


def test_pattern_mismatch_is_reported() -> None:
    registry = ToolRegistry.from_file("data/tools.json")
    validator = SchemaValidator()

    issues = validator.validate_call(registry.get("get_balance"), {"customer_id": "1001"}, "get_balance")

    assert [issue.code for issue in issues] == ["pattern_mismatch"]
