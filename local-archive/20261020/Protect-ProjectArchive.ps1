#requires -Version 7.5
# Dot-source for authenticated, chunked AES-256-GCM backup encryption.
if(-not('ProjectArchiveCrypto'-as[type])){
 Add-Type -TypeDefinition @'
using System;using System.IO;using System.Text;using System.Security.Cryptography;
public static class ProjectArchiveCrypto {
 static readonly byte[] Magic=Encoding.ASCII.GetBytes("KSAR0001");
 static byte[] Read(Stream s,int n){byte[] b=new byte[n];s.ReadExactly(b);return b;}
 static byte[] Nonce(byte[] prefix,uint counter){byte[] n=new byte[12];prefix.CopyTo(n,0);BitConverter.GetBytes(counter).CopyTo(n,8);return n;}
 public static void Encrypt(string input,string output,byte[] key){
  byte[] prefix=RandomNumberGenerator.GetBytes(8),buffer=new byte[1048576];uint counter=0;
  using(var src=File.OpenRead(input))using(var dst=new FileStream(output,FileMode.CreateNew))using(var aes=new AesGcm(key,16)){
   dst.Write(Magic);dst.Write(prefix);
   while(true){int count=0,n;while(count<buffer.Length&&(n=src.Read(buffer,count,buffer.Length-count))>0)count+=n;
    byte[] cipher=new byte[count],tag=new byte[16];aes.Encrypt(Nonce(prefix,counter++),buffer.AsSpan(0,count),cipher,tag,Magic);dst.Write(BitConverter.GetBytes(count));dst.Write(cipher);dst.Write(tag);
    if(count==0)break;if(counter==uint.MaxValue)throw new Exception("Archive counter overflow");
   }
  }
 }
 public static string Verify(string input,byte[] key,string restore){
  using(var src=File.OpenRead(input))using(var aes=new AesGcm(key,16))using(var hash=IncrementalHash.CreateHash(HashAlgorithmName.SHA256)){
   var magic=Read(src,8);if(!magic.AsSpan().SequenceEqual(Magic))throw new Exception("Invalid encrypted archive header");var prefix=Read(src,8);uint counter=0;
   using(var dst=string.IsNullOrEmpty(restore)?null:new FileStream(restore,FileMode.CreateNew)){
    while(true){int count=BitConverter.ToInt32(Read(src,4));if(count<0||count>1048576)throw new Exception("Invalid encrypted chunk length");byte[] cipher=Read(src,count),tag=Read(src,16),plain=new byte[count];aes.Decrypt(Nonce(prefix,counter++),cipher,tag,plain,Magic);hash.AppendData(plain);if(dst!=null)dst.Write(plain);if(count==0){if(src.Position!=src.Length)throw new Exception("Unexpected trailing archive bytes");break;}if(counter==uint.MaxValue)throw new Exception("Archive counter overflow");
    }
   }
   return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
  }
 }
}
'@
}
function Get-ProjectArchiveKey([string]$Project){
 $dir='C:\Users\SSAFY\Documents\StorageCleanupRetained\KraStrategyKeys-20261020'
 [void][IO.Directory]::CreateDirectory($dir)
 $acl=[IO.FileSystemAclExtensions]::GetAccessControl([IO.DirectoryInfo]::new($dir),[Security.AccessControl.AccessControlSections]::Access);$acl.SetAccessRuleProtection($true,$false)
 $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User
 $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($sid,'FullControl','ContainerInherit,ObjectInherit','None','Allow'))
 [IO.FileSystemAclExtensions]::SetAccessControl([IO.DirectoryInfo]::new($dir),$acl)
 $path=Join-Path $dir ($Project+'.key.json')
 if(-not[IO.File]::Exists($path)){
  $key=[Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
  [pscustomobject]@{Schema='project-archive-key/v1';Project=$Project;Cipher='AES-256-GCM/KSAR0001';KeyBase64=[Convert]::ToBase64String($key);CreatedUtc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath $path -Encoding utf8
 }
 $json=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
 if($json.Project-cne$Project){throw 'Encryption key identity mismatch'}
 return ,([Convert]::FromBase64String($json.KeyBase64))
}
function Protect-ProjectPayload([string]$Path,[string]$Project){
 $target=$Path+'.aes';$proof=$target+'.verified.json';$key=Get-ProjectArchiveKey $Project
 $sourceHash=(Get-FileHash -LiteralPath $Path).Hash.ToLowerInvariant()
 if([IO.File]::Exists($target)){
  if(-not(Test-Path $proof)){Remove-Item -LiteralPath $target -Force}
  else{$prior=Get-Content $proof -Raw|ConvertFrom-Json;if($prior.PlaintextSha256-cne$sourceHash-or(Get-FileHash $target).Hash.ToLowerInvariant()-cne$prior.Sha256){throw 'Previous encrypted payload changed'};return $prior}
 }
 [ProjectArchiveCrypto]::Encrypt($Path,$target,$key)
 $h=[ProjectArchiveCrypto]::Verify($target,$key,$null)
 if($h-cne$sourceHash){throw 'Encrypted backup restore hash mismatch'}
 $result=[pscustomobject]@{Path=$target;PlaintextSha256=$sourceHash;Sha256=(Get-FileHash $target).Hash.ToLowerInvariant();Bytes=(Get-Item $target).Length;Cipher='AES-256-GCM/KSAR0001';AuthenticatedRestoreVerified=$true}
 $result|ConvertTo-Json|Set-Content $proof -Encoding utf8
 return $result
}
