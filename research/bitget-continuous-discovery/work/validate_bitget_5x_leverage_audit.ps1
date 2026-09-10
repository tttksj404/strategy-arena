param(
    [Parameter(Mandatory = $true)]
    [string]$Report
)

$ErrorActionPreference = 'Stop'

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw "VALIDATION_FAILED: $Message"
    }
}

Assert-Condition (Test-Path -LiteralPath $Report) "report does not exist: $Report"
$payload = Get-Content -Raw -LiteralPath $Report | ConvertFrom-Json
Assert-Condition ($payload.mode -eq 'public-api-paper-only') 'mode must remain paper-only'
Assert-Condition (@($payload.live_promotable).Count -eq 0) 'live promotion list must stay empty'
Assert-Condition (@($payload.candidates).Count -gt 0) 'no 5x candidate was audited'
foreach ($candidate in @($payload.candidates)) {
    Assert-Condition ([double]$candidate.max_leverage -eq 5.0) "$($candidate.candidate) is not a 5x-cap candidate"
    Assert-Condition ([double]$candidate.max_realized_leverage -gt 0.0 -and [double]$candidate.max_realized_leverage -le 5.000001) "$($candidate.candidate) has invalid realized leverage"
    Assert-Condition ($candidate.paper_pass -eq $true -and $candidate.market_gate -eq $true) "$($candidate.candidate) lost its current paper gate"
    Assert-Condition ($candidate.simulated_liquidation -eq $false) "$($candidate.candidate) triggered a liquidation surrogate"
    Assert-Condition ([double]$candidate.base_return -gt 0.0 -and [double]$candidate.stressed_return -gt 0.0) "$($candidate.candidate) has nonpositive return"
    Assert-Condition ([double]$candidate.max_drawdown -le 0.25) "$($candidate.candidate) exceeds drawdown gate"
    Assert-Condition ([int]$candidate.positive_blocks -ge 3) "$($candidate.candidate) lacks positive blocks"
}
"BITGET_5X_AUDIT_VALIDATION_PASS candidates=$(@($payload.candidates).Count) actual_five_x=$(@($payload.candidates | Where-Object { $_.max_realized_leverage -ge 4.999999 }).Count)"
