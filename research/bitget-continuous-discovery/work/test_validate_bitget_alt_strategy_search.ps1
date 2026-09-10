$validator = Join-Path $PSScriptRoot 'validate_bitget_alt_strategy_search.ps1'
$source = Get-Content -LiteralPath $validator -Raw

if ($source -match 'ConvertFrom-Json\s+-Depth') {
    throw 'Validator passes -Depth to ConvertFrom-Json, which Windows PowerShell 5.1 does not support.'
}
