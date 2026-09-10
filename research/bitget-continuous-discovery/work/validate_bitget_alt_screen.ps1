param(
    [Parameter(Mandatory = $true)]
    [string]$Report
)

$ErrorActionPreference = 'Stop'

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

Require (Test-Path -LiteralPath $Report -PathType Leaf) 'screen report is missing'
$payload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
Require ($payload.mode -eq 'public-api-paper-only') 'screen is not public-api paper-only'
Require (@($payload.live_promotable).Count -eq 0) 'screen exposes a live promotion'
$intervals = @($payload.results)
Require ($intervals.Count -eq 14) 'all supported intervals were not screened'
$expectedResults = @($intervals[0].results).Count
Require ($expectedResults -gt 0) 'screen has no candidate rows'

$records = 0
$passes = 0
foreach ($interval in $intervals) {
    $rows = @($interval.results)
    Require ($rows.Count -eq $expectedResults) "candidate count differs for $($interval.interval)"
    foreach ($row in $rows) {
        $records += 1
        if ($row.exchange_screen_pass) {
            $passes += 1
            Require ($row.market_gate -eq $true) "passed row failed current market gate: $($interval.interval)"
            Require ($row.simulated_liquidation -eq $false) "passed row has simulated liquidation: $($interval.interval)"
            Require ($row.promotion -eq 'paper-only') "passed row is not paper-only: $($interval.interval)"
            Require ([double]$row.oos_days -ge 90.0) "passed row has insufficient OOS duration: $($interval.interval)"
            Require ([double]$row.test.net_return -gt 0.0) "passed row has nonpositive base return: $($interval.interval)"
            Require ([double]$row.stress_return -gt 0.0) "passed row has nonpositive stress return: $($interval.interval)"
            Require ([double]$row.test.max_drawdown -le 0.25) "passed row breaches drawdown limit: $($interval.interval)"
            Require ([int]$row.test.positive_blocks -ge 3) "passed row lacks positive blocks: $($interval.interval)"
        }
    }
}

Write-Output "BITGET_SCREEN_VALIDATION_PASS intervals=$($intervals.Count) candidates_per_interval=$expectedResults records=$records passes=$passes"
