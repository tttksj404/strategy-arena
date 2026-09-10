[CmdletBinding()]
param(
    [string]$SourceRoot = '',
    [Parameter(Mandatory = $true)][string]$RepoRoot
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($SourceRoot)) { $SourceRoot = Split-Path -Parent $PSScriptRoot }
$taskSource = (Resolve-Path -LiteralPath $SourceRoot).Path
$taskRepo = (Resolve-Path -LiteralPath $RepoRoot).Path
$taskDestination = [IO.Path]::GetFullPath((Join-Path $taskRepo 'research\bitget-continuous-discovery'))
if (-not $taskDestination.StartsWith($taskRepo + [IO.Path]::DirectorySeparatorChar)) { throw 'Invalid bundle target' }
$records = [Collections.Generic.List[object]]::new()
function Copy-EvidenceFile {
    param([string]$InputPath, [string]$RelativePath, [bool]$Compress = $false)
    $original = Get-Item -LiteralPath $InputPath
    $storedRelative = $RelativePath + $(if ($Compress) { '.gz' } else { '' })
    $target = [IO.Path]::GetFullPath((Join-Path $taskDestination $storedRelative))
    if (-not $target.StartsWith($taskDestination + [IO.Path]::DirectorySeparatorChar)) { throw 'Invalid file target' }
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force
    $originalHash = (Get-FileHash -LiteralPath $original.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Compress) {
        $inputStream = [IO.File]::OpenRead($original.FullName)
        $outputStream = [IO.File]::Create($target)
        $zip = [IO.Compression.GZipStream]::new($outputStream, [IO.Compression.CompressionLevel]::Optimal)
        try { $inputStream.CopyTo($zip) } finally { $zip.Dispose(); $outputStream.Dispose(); $inputStream.Dispose() }
        $storedStream = [IO.File]::OpenRead($target)
        $unzipped = [IO.Compression.GZipStream]::new($storedStream, [IO.Compression.CompressionMode]::Decompress)
        $hasher = [Security.Cryptography.SHA256]::Create()
        try { $roundTripHash = [BitConverter]::ToString($hasher.ComputeHash($unzipped)).Replace('-', '').ToLowerInvariant() }
        finally { $hasher.Dispose(); $unzipped.Dispose(); $storedStream.Dispose() }
    } else {
        Copy-Item -LiteralPath $original.FullName -Destination $target -Force
        $roundTripHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    if ($roundTripHash -ne $originalHash) { throw ('Content mismatch: ' + $RelativePath) }
    $stored = Get-Item -LiteralPath $target
    if ($stored.Length -ge 95000000) { throw ('Bundle file too large: ' + $storedRelative) }
    $records.Add([pscustomobject]@{path=$storedRelative.Replace('\','/'); original_bytes=$original.Length; original_sha256=$originalHash; stored_bytes=$stored.Length; stored_sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant(); compression=$(if ($Compress) {'gzip'} else {'none'})})
}
$taskWork = Join-Path $taskSource 'work'
$files = Get-ChildItem -LiteralPath $taskWork -File | Where-Object { $_.Extension -in @('.py','.pyi','.ps1','.json','.md','.lock') }
foreach ($file in $files) { Copy-EvidenceFile $file.FullName ('work\' + $file.Name) ($file.Extension -eq '.json' -and $file.Length -gt 10000000) }
$stubRoot = Join-Path $taskWork 'type-stubs'
if (Test-Path -LiteralPath $stubRoot) {
    foreach ($file in (Get-ChildItem -LiteralPath $stubRoot -Recurse -File | Where-Object { $_.Extension -in @('.pyi','.md') })) {
        $suffix = $file.FullName.Substring($stubRoot.Length).TrimStart([IO.Path]::DirectorySeparatorChar)
        Copy-EvidenceFile $file.FullName ('work\type-stubs\' + $suffix)
    }
}
foreach ($folderName in @('x_discovery_20260910','x_volume_control_20260910','x_discovery_20260910_recheck','x_volume_control_20260910_recheck','x_discovery_evidence')) {
    $folderPath = Join-Path $taskWork $folderName
    if (Test-Path -LiteralPath $folderPath) {
        foreach ($file in (Get-ChildItem -LiteralPath $folderPath -File)) {
            if ($folderName -eq 'x_discovery_evidence' -and $file.Extension -eq '.log' -and -not (Test-Path -LiteralPath ([IO.Path]::ChangeExtension($file.FullName, 'meta.json')))) { continue }
            Copy-EvidenceFile $file.FullName ('work\' + $folderName + '\' + $file.Name)
        }
    }
}
$refreshRoot = Join-Path $taskWork 'bitget_history_refresh_20260910'
if (Test-Path -LiteralPath $refreshRoot) {
    foreach ($file in (Get-ChildItem -LiteralPath $refreshRoot -Recurse -File)) {
        $suffix = $file.FullName.Substring($refreshRoot.Length).TrimStart([IO.Path]::DirectorySeparatorChar)
        Copy-EvidenceFile $file.FullName ('data-refresh\' + $suffix)
    }
}
$dailyName = 'bitget_history_full_daily_20260824_v2'
$dailyRoot = Join-Path $taskWork $dailyName
foreach ($file in (Get-ChildItem -LiteralPath $dailyRoot -Recurse -File)) {
    $suffix = $file.FullName.Substring($dailyRoot.Length).TrimStart([IO.Path]::DirectorySeparatorChar)
    Copy-EvidenceFile $file.FullName ('work\' + $dailyName + '\' + $suffix)
}
foreach ($folder in (Get-ChildItem -LiteralPath $taskWork -Directory -Filter 'bitget_history_*')) {
    $manifest = Join-Path $folder.FullName 'MANIFEST.json'
    if (Test-Path -LiteralPath $manifest) { Copy-EvidenceFile $manifest ('data-manifests\' + $folder.Name + '\MANIFEST.json') }
}
foreach ($file in (Get-ChildItem -LiteralPath $taskWork -File -Filter '*.log')) { Copy-EvidenceFile $file.FullName ('work\evidence\legacy\' + $file.Name) }
foreach ($name in @('strategy-arena-top-return-candidates-ko.pdf','strategy-arena-x-source-experiments-ko.pdf')) {
    $pdf = Join-Path $taskSource ('output\pdf\' + $name)
    if (Test-Path -LiteralPath $pdf) { Copy-EvidenceFile $pdf ('output\pdf\' + $name) }
}
$bundleManifest = [pscustomobject]@{schema='strategy-arena-research-bundle-v1'; files=$records; all_roundtrip_hashes_verified=$true; excludes=@('Older intraday raw candle archives: metadata and collectors only; locked daily raw archive and recent update included','Full external X article text and media','Secrets and unrelated dirty repository changes')}
$manifestPath = Join-Path $taskDestination 'BUNDLE_MANIFEST.json'
[IO.File]::WriteAllText($manifestPath, ($bundleManifest | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
[pscustomobject]@{status='BUNDLE_VERIFIED'; files=$records.Count; stored_bytes=($records|Measure-Object stored_bytes -Sum).Sum; manifest_sha256=(Get-FileHash -LiteralPath $manifestPath).Hash} | ConvertTo-Json -Compress
