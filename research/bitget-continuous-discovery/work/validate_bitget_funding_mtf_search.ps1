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
$payload = Get-Content -Raw -LiteralPath $Report | ConvertFrom-Json
$expectedIntervals = @('1H', '2H', '4H', '6H', '12H', '1D')

Assert-Condition ($payload.mode -eq 'public-api-paper-only') 'report mode must remain paper-only'
Assert-Condition ($payload.short_horizon_nonpromotable -eq $true) 'short-horizon nonpromotion flag is missing'
Assert-Condition (@($payload.live_promotable).Count -eq 0) 'live promotion list must be empty'
Assert-Condition (@($payload.intervals).Count -eq $expectedIntervals.Count) 'interval count does not match contract'
Assert-Condition ((@($payload.intervals.interval) -join ',') -eq ($expectedIntervals -join ',')) 'interval order does not match contract'
Assert-Condition ((@($payload.source_endpoints) -join ',') -eq '/api/v2/mix/market/candles,/api/v2/mix/market/history-fund-rate,/api/v2/mix/market/tickers') 'unexpected public endpoint set'

$totalConfigurations = 0
$totalWinners = 0
$shortPasses = 0
$capFive = 0
$actualFive = 0

foreach ($interval in @($payload.intervals)) {
    Assert-Condition ([int]$interval.configurations -eq 648) "$($interval.interval) configuration count must be 648"
    Assert-Condition ([double]$interval.observation_days -gt 0) "$($interval.interval) has no observed bars"
    Assert-Condition (@($interval.funding_event_counts.psobject.Properties).Count -eq 13) "$($interval.interval) does not cover all 13 symbols"
    foreach ($entry in @($interval.funding_event_counts.psobject.Properties)) {
        Assert-Condition ([int]$entry.Value -gt 0) "$($interval.interval) has no funding events for $($entry.Name)"
    }
    $totalConfigurations += [int]$interval.configurations
    foreach ($winner in @($interval.validation_winners)) {
        Assert-Condition ($winner.promotion -eq 'paper-only') "winner $($winner.candidate) is not paper-only"
        Assert-Condition (@(1.0, 2.0, 3.0, 5.0) -contains [double]$winner.max_leverage) "winner $($winner.candidate) has an undeclared leverage cap"
        Assert-Condition ([double]$winner.max_realized_leverage -le ([double]$winner.max_leverage + 0.000001)) "winner $($winner.candidate) exceeds its leverage cap"
        $totalWinners += 1
        if ([double]$winner.max_leverage -eq 5.0) {
            $capFive += 1
        }
        if ([double]$winner.max_realized_leverage -ge 4.999999) {
            $actualFive += 1
        }
        if ($winner.short_horizon_paper_pass -eq $true) {
            Assert-Condition ($winner.simulated_liquidation -eq $false) "short pass $($winner.candidate) has a liquidation surrogate"
            Assert-Condition ([double]$winner.locked_test.net_return -gt 0) "short pass $($winner.candidate) has nonpositive locked-test return"
            Assert-Condition ([double]$winner.locked_test_stress.net_return -gt 0) "short pass $($winner.candidate) has nonpositive stressed locked-test return"
            Assert-Condition ([double]$winner.locked_test.max_drawdown -le 0.25) "short pass $($winner.candidate) exceeds drawdown gate"
            Assert-Condition ([int]$winner.locked_test.positive_blocks -ge 3) "short pass $($winner.candidate) lacks positive blocks"
            $shortPasses += 1
        }
    }
}

Assert-Condition ($totalConfigurations -eq 3888) 'total configurations must be 3888'
"BITGET_FUNDING_MTF_VALIDATION_PASS configurations=$totalConfigurations winners=$totalWinners short_passes=$shortPasses cap5=$capFive actual5=$actualFive"
