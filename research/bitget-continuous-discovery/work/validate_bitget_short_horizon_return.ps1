param(
    [Parameter(Mandatory = $true)]
    [string]$Report,
    [string]$ResultPath = ''
)

$ErrorActionPreference = 'Stop'
$defaultResultPath = Join-Path $PSScriptRoot '..\.omx\specs\autoresearch-bitget-short-horizon-return\result.json'
$resultPath = if ($ResultPath) { $ResultPath } else { $defaultResultPath }

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

try {
    Require (Test-Path -LiteralPath $Report) 'report is missing'
    $payload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    $supportedModes = @(
        'public-data-paper-only-fixed-notional-counterfactual-no-liquidation-stop',
        'public-data-paper-only-classic-fixed-notional-counterfactual-no-liquidation-stop'
    )
    Require ($payload.mode -in $supportedModes) 'mode does not preserve the fixed-notional counterfactual research boundary'
    Require ($payload.research_only -eq $true) 'report is not research-only'
    Require ($payload.live_promotable -eq $false) 'report exposes a live promotion'
    Require ($payload.contract.liquidation_stop_disabled -eq $true) 'report does not disclose the no-stop calculation'
    Require ($payload.contract.account_equity_compounding -eq $false) 'report does not disclose the non-compounding PnL basis'
    Require ($payload.contract.fixed_notional_per_bar -eq $true) 'report does not disclose the fixed-notional PnL basis'
    Require ($payload.contract.pnl_unit -eq 'one initial-capital notional redeployed each bar; not an account return') 'report does not disclose the PnL unit'
    Require ($payload.contract.not_an_executable_account_equity_path -eq $true) 'report can be mistaken for an executable equity path'
    Require ($payload.contract.test_evaluated_after_validation_selection -eq $true) 'selection/test sequencing is absent'
    Require ($payload.contract.base_fee_per_side -eq 0.0006) 'base fee contract mismatch'
    Require ($payload.contract.base_slippage_per_turnover -eq 0.0003) 'base slippage contract mismatch'
    Require ($payload.data.integrity_verified -eq $true) 'input integrity is not verified'
    Require ($payload.predeclared_search.raw_configurations -gt 0) 'search grid is empty'
    Require ($null -ne $payload.selected_validation_winner) 'validation selection is missing'
    Require ($null -ne $payload.locked_test_result) 'locked test result is missing'
    Require ($payload.locked_test_result.live_promotable -eq $false) 'locked result exposes a live promotion'
    $result = [ordered]@{
        status = 'passed'
        passed = $true
        summary = "validated $($payload.predeclared_search.raw_configurations) counterfactual configurations; live promotions=0"
        report = (Resolve-Path -LiteralPath $Report).Path
    }
    $result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resultPath -Encoding utf8
    Write-Output "SHORT_HORIZON_RETURN_VALIDATION_PASS candidates=$($payload.predeclared_search.raw_configurations)"
} catch {
    $result = [ordered]@{
        status = 'failed'
        passed = $false
        summary = $_.Exception.Message
    }
    $result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resultPath -Encoding utf8
    throw
}
