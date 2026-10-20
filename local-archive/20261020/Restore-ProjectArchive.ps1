#requires -Version 7.5
param([Parameter(Mandatory)][string]$Manifest,[Parameter(Mandatory)][string]$Destination,[string]$KeyFile)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'Protect-ProjectArchive.ps1')
$m=Get-Content -LiteralPath $Manifest -Raw|ConvertFrom-Json -DateKind String
$root=[IO.Path]::GetFullPath($Destination);[void][IO.Directory]::CreateDirectory($root)
$downloads=Join-Path $root '.download';[void][IO.Directory]::CreateDirectory($downloads)
$key=if($KeyFile){[Convert]::FromBase64String((Get-Content -LiteralPath $KeyFile -Raw|ConvertFrom-Json).KeyBase64)}else{$null}
$old=$env:GH_TOKEN
try{
 if(-not$env:GH_TOKEN){$env:GIT_TERMINAL_PROMPT='0';$env:GCM_INTERACTIVE='never';$c="protocol=https`nhost=github.com`n`n"|& git credential fill 2>$null;if($LASTEXITCODE-eq0){foreach($l in $c){if($l.StartsWith('password=')){$env:GH_TOKEN=$l.Substring(9)}}}}
 foreach($a in $m.Assets){
  $path=Join-Path $downloads $a.Name
  if(-not(Test-Path $path)){& gh api "repos/$($m.Repository)/releases/assets/$($a.AssetId)?nonce=$([DateTime]::UtcNow.Ticks)" -H 'Accept: application/octet-stream' > $path;if($LASTEXITCODE-ne0){throw 'Asset download failed'}}
  if((Get-Item $path).Length-ne$a.Bytes-or(Get-FileHash $path).Hash.ToLowerInvariant()-cne$a.Sha256){throw 'Downloaded asset hash mismatch'}
  $payload=$path;$private=@($m.EncryptedPayloads|Where-Object {$_.Asset.AssetId-eq$a.AssetId})
  if($private.Count){if(-not$key){throw 'Key file is required to restore encrypted assets'};$payload=$path.Substring(0,$path.Length-4);if(Test-Path $payload){throw 'Decrypted output already exists'};$h=[ProjectArchiveCrypto]::Verify($path,$key,$payload);if($h-cne$private[0].PlaintextSha256){throw 'Decrypted payload hash mismatch'}}
  if($payload.EndsWith('.zip',[StringComparison]::OrdinalIgnoreCase)){
   $extractRoot=if($a.Name-like'git-metadata*'){Join-Path $root '.original-git-metadata'}else{$root};[void][IO.Directory]::CreateDirectory($extractRoot)
  }
  if($payload.EndsWith('.zip',[StringComparison]::OrdinalIgnoreCase)){
   $zip=[IO.Compression.ZipFile]::OpenRead($payload)
   try{foreach($e in $zip.Entries){$dest=[IO.Path]::GetFullPath((Join-Path $extractRoot $e.FullName));if(-not$dest.StartsWith($extractRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'ZIP path outside destination'};if($e.FullName.EndsWith('/')){[void][IO.Directory]::CreateDirectory($dest);continue};[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($dest));[IO.Compression.ZipFileExtensions]::ExtractToFile($e,$dest,$false)}}finally{$zip.Dispose()}
  }elseif($payload.EndsWith('.bundle')){Copy-Item -LiteralPath $payload -Destination (Join-Path $root 'history.bundle') -ErrorAction Stop}
 }
 foreach($f in $m.Files){$p=Join-Path $root $f.Relative;if(-not(Test-Path -LiteralPath $p)-or(Get-Item -LiteralPath $p).Length-ne$f.Bytes-or(Get-FileHash -LiteralPath $p).Hash.ToLowerInvariant()-cne$f.Sha256){throw 'Restored file hash mismatch'}}
 Write-Output "PROJECT_ARCHIVE_RESTORE_VERIFIED files=$($m.Files.Count)"
}finally{$env:GH_TOKEN=$old;$c=$null;$key=$null}
