from src.executor.mock_telco_api import MockTelcoApi
from src.registry.contract_registry import ContractRegistry
from src.validation.contract_checker import ContractChecker


def test_active_verified_customer_passes_data_package_contract() -> None:
    contracts = ContractRegistry.from_file("data/tool_contracts.json")
    executor = MockTelcoApi.from_file("data/mock_telco_db.json")
    checker = ContractChecker()

    issues = checker.check(
        contracts.get("add_data_package"),
        executor.get_subscriber("C001"),
        {"customer_id": "C001", "package_code": "DATA10"},
        customer_verified=True,
    )

    assert issues == []


def test_unverified_customer_fails_sensitive_write_contract() -> None:
    contracts = ContractRegistry.from_file("data/tool_contracts.json")
    executor = MockTelcoApi.from_file("data/mock_telco_db.json")
    checker = ContractChecker()

    issues = checker.check(
        contracts.get("suspend_line"),
        executor.get_subscriber("C001"),
        {"customer_id": "C001", "reason": "customer_request"},
        customer_verified=False,
    )

    assert any(issue.code == "permission_denied" for issue in issues)


def test_suspended_customer_fails_data_package_contract() -> None:
    contracts = ContractRegistry.from_file("data/tool_contracts.json")
    executor = MockTelcoApi.from_file("data/mock_telco_db.json")
    checker = ContractChecker()

    issues = checker.check(
        contracts.get("add_data_package"),
        executor.get_subscriber("C002"),
        {"customer_id": "C002", "package_code": "DATA10"},
        customer_verified=True,
    )

    assert [issue.code for issue in issues] == ["precondition_failed"]
    assert "active" in issues[0].message
