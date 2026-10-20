#requires -Version 7.5
param([Parameter(Mandatory)][string]$KeyFile,[Parameter(Mandatory)][string]$Destination,[string]$EncryptedPath=(Join-Path $PSScriptRoot 'supplemental.zip.aes'))
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'Protect-ProjectArchive.ps1')
$m=Get-Content (Join-Path $PSScriptRoot 'SUPPLEMENTAL.json') -Raw|ConvertFrom-Json
if((Get-FileHash $EncryptedPath).Hash.ToLowerInvariant()-cne$m.CipherSha256){throw 'Supplemental ciphertext hash mismatch'}
$root=[IO.Path]::GetFullPath($Destination);[void][IO.Directory]::CreateDirectory($root)
$key=[Convert]::FromBase64String((Get-Content $KeyFile -Raw|ConvertFrom-Json).KeyBase64)
$zip=Join-Path $root 'supplemental.restore.zip'
if([ProjectArchiveCrypto]::Verify($EncryptedPath,$key,$zip)-cne$m.PlaintextSha256){throw 'Supplemental authenticated restoration mismatch'}
$archive=[IO.Compression.ZipFile]::OpenRead($zip)
try{foreach($e in $archive.Entries){$dest=[IO.Path]::GetFullPath((Join-Path $root $e.FullName));if(-not$dest.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Supplemental path escaped destination'};[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($dest));[IO.Compression.ZipFileExtensions]::ExtractToFile($e,$dest,$false)}}finally{$archive.Dispose()}
foreach($f in $m.Files){$p=Join-Path $root $f.Relative;if((Get-FileHash $p).Hash.ToLowerInvariant()-cne$f.Sha256){throw 'Supplemental restored file hash mismatch'}}
Write-Output "SUPPLEMENTAL_RESTORE_VERIFIED files=$($m.Files.Count)"
