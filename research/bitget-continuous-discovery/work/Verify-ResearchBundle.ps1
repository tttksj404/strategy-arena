[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BundleRoot,
    [Parameter(Mandatory = $true)][string]$CredentialFile,
    [string]$EvidencePath = ''
)
$ErrorActionPreference = 'Stop'
$bundle = (Resolve-Path -LiteralPath $BundleRoot).Path
$manifestPath = Join-Path $bundle 'BUNDLE_MANIFEST.json'
$manifest = [IO.File]::ReadAllText($manifestPath) | ConvertFrom-Json
$credentialText = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $CredentialFile).Path)
$credentialTokens = @([regex]::Matches($credentialText, '[A-Za-z0-9_-]{16,}') | ForEach-Object { $_.Value } | Select-Object -Unique)
$credentialText = $null
if ($credentialTokens.Count -lt 2) { throw 'Credential scan input could not be safely tokenized; no values are printed.' }
$verified = 0
foreach ($entry in $manifest.files) {
    $path = [IO.Path]::GetFullPath((Join-Path $bundle $entry.path))
    if (-not $path.StartsWith($bundle + [IO.Path]::DirectorySeparatorChar)) { throw 'Unsafe manifest path' }
    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $entry.stored_sha256) { throw ('Stored evidence hash mismatch: ' + $entry.path) }
    $verified++
}
$scanned = 0
$violations = [Collections.Generic.List[string]]::new()
foreach ($file in (Get-ChildItem -LiteralPath $bundle -Recurse -File)) {
    $raw = [IO.File]::OpenRead($file.FullName)
    $stream = $raw
    if ($file.Extension -eq '.gz') { $stream = [IO.Compression.GZipStream]::new($raw, [IO.Compression.CompressionMode]::Decompress) }
    try {
        $buffer = New-Object byte[] 65536
        $tail = ''
        while (($count = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $segment = $tail + [Text.Encoding]::UTF8.GetString($buffer, 0, $count)
            $knownMatch = $false
            foreach ($token in $credentialTokens) { if ($segment.Contains($token)) { $knownMatch = $true; break } }
            $genericMatch = $segment -match '(?i)gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'
            if ($knownMatch -or $genericMatch) { $violations.Add($file.FullName.Substring($bundle.Length)); break }
            $tail = $segment.Substring([Math]::Max(0, $segment.Length - 512))
        }
        $scanned++
    } finally { $stream.Dispose(); if ($stream -ne $raw) { $raw.Dispose() } }
}
$credentialTokens = $null
if ($violations.Count -gt 0) { throw ('Potential secret material in bundle files: ' + ($violations -join ', ')) }
$evidence = [pscustomobject]@{status='BUNDLE_SECURITY_VERIFIED'; manifest_entries_verified=$verified; files_scanned=$scanned; known_credential_matches=0; generic_key_pattern_matches=0; scan_scope='All bundle bytes and decompressed gzip; no credential values logged'; manifest_sha256=(Get-FileHash -LiteralPath $manifestPath).Hash.ToLowerInvariant()}
$json = $evidence | ConvertTo-Json -Compress
if ($EvidencePath) { [IO.File]::WriteAllText([IO.Path]::GetFullPath($EvidencePath), $json, [Text.UTF8Encoding]::new($false)) }
$json
