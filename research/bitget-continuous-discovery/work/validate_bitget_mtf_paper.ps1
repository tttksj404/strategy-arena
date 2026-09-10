param(
    [Parameter(Mandatory = $true)]
    [string]$Report
)

$ErrorActionPreference = 'Stop'

function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not $Condition) {
        throw "VALIDATION_FAILED: $Message"
    }
}

Assert-Condition (Test-Path -LiteralPath $Report) "report does not exist: $Report"
$raw = Get-Content -Raw -LiteralPath $Report
Assert-Condition ($raw -notmatch 'NaN|Infinity') 'report contains a non-finite JSON number'
$payload = $raw | ConvertFrom-Json
$expectedIntervals = @('1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '3D', '1W', '1M')

Assert-Condition ($payload.mode -eq 'public-api-paper-only') 'report mode must remain paper-only'
Assert-Condition (@($payload.results).Count -eq $expectedIntervals.Count) 'result count does not match interval contract'
Assert-Condition ((@($payload.results.interval) -join ',') -eq ($expectedIntervals -join ',')) 'interval order does not match contract'

$liquidationIntervals = 0
foreach ($result in @($payload.results)) {
    Assert-Condition ($result.regime_on -is [bool]) "$($result.interval) regime_on is not a JSON boolean"
    Assert-Condition ([double]$result.leverage -ge 0.0 -and [double]$result.leverage -le 5.0) "$($result.interval) leverage is outside declared bounds"
    Assert-Condition (-not [double]::IsNaN([double]$result.test.net_return)) "$($result.interval) test return is NaN"
    Assert-Condition (-not [double]::IsInfinity([double]$result.test.net_return)) "$($result.interval) test return is infinite"
    Assert-Condition ([int]$result.simulated_liquidations -ge 0) "$($result.interval) has an invalid liquidation count"
    Assert-Condition (@($result.gate_reasons).Count -gt 0) "$($result.interval) lacks its required paper-only rejection reason"
    if ([int]$result.simulated_liquidations -gt 0) {
        $liquidationIntervals += 1
    }
}

"BITGET_MTF_PAPER_VALIDATION_PASS intervals=$(@($payload.results).Count) passing_intervals=0 liquidation_intervals=$liquidationIntervals"
