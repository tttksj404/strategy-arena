param(
    [Parameter(Mandatory = $true)]
    [string]$Report
)

$ErrorActionPreference = 'Stop'
$resultPath = Join-Path $PSScriptRoot '..\.omx\specs\autoresearch-bitget-alt-strategy-search\result.json'

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

try {
    Require (Test-Path -LiteralPath $Report) 'report is missing'
    $payload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    Require ($payload.mode -eq 'paper-only-no-live-orders') 'mode is not paper-only'
    Require ($payload.grid.total_configurations -ge 100000) 'candidate grid is below liquidity-expanded search threshold'
    Require ($payload.grid.liquidity_floors.Count -eq 4) 'liquidity expansion is missing an expected floor'
    Require ($payload.validation_records.Count -eq $payload.grid.total_configurations) 'validation record count mismatch'
    Require ($payload.contract.test_evaluated_after_validation_selection -eq $true) 'selection/test sequencing is absent'
    Require ($payload.contract.base_fee_per_side -eq 0.0006) 'base fee contract mismatch'
    Require ($payload.contract.base_slippage_per_turnover -eq 0.0003) 'base slippage contract mismatch'
    Require ($payload.contract.stress_slippage_per_turnover -eq 0.0006) 'stress slippage contract mismatch'
    Require ($payload.data.integrity_verified -eq $true) 'input integrity is not verified'
    Require ($payload.data.source_hashes.Count -eq 13) 'source hash count mismatch'
    Require ($payload.locked_test_results.Count -eq $payload.validation_winners.Count) 'locked test was not limited to validation winners'
    Require ($payload.live_promotable.Count -eq 0) 'report exposes a live promotion'
    foreach ($record in $payload.validation_records) {
        Require (-not ($record.PSObject.Properties.Name -contains 'test')) "validation record leaked a test metric: $($record.id)"
    }
    foreach ($result in $payload.locked_test_results) {
        Require ($result.live_promotable -eq $false) "locked test exposed a live promotion: $($result.id)"
    }
    $result = [ordered]@{
        status = 'passed'
        passed = $true
        summary = "validated $($payload.grid.total_configurations) configurations; locked tests=$($payload.locked_test_results.Count); live promotions=0"
        report = (Resolve-Path -LiteralPath $Report).Path
    }
    $result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resultPath -Encoding utf8
    Write-Output "BROAD_SEARCH_VALIDATION_PASS candidates=$($payload.grid.total_configurations) winners=$($payload.validation_winners.Count)"
} catch {
    $result = [ordered]@{
        status = 'failed'
        passed = $false
        summary = $_.Exception.Message
    }
    $result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resultPath -Encoding utf8
    throw
}
